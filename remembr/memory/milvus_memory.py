from dataclasses import dataclass, asdict

import datetime, time
import uuid
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem
from remembr.memory.memory_policy import MemoryPolicy, MemoryRecord, PolicyResult

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970


def is_local_milvus(address) -> bool:
    # A db_ip like "milvus_demo.db" refers to a Milvus Lite local file
    # rather than a server address.
    return str(address).endswith('.db')


class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        if is_local_milvus(address):
            # Milvus Lite: an embedded, file-backed Milvus (no docker needed)
            connections.connect(uri=address)
        else:
            connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)

        if utility.has_collection(collection_name):
            # Open existing collections with their stored schema so databases
            # created before newer fields (e.g. camera_id) keep working.
            collection = Collection(name=collection_name)
        else:
            fields = [
                FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
                FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
                FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
                FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
                FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
                FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),
                FieldSchema(name='camera_id', dtype=DataType.VARCHAR, description='camera that produced the caption', max_length=100),
            ]
            schema = CollectionSchema(fields=fields, description='text image search')
            collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            # expr="id > 3",
            output_fields=["id", "text_embedding"]
        )

        return res




class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT,
                 policy: MemoryPolicy = None, prune_every: int = 100):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset

        # Optional lifelong memory management: when a policy is set, it is
        # applied automatically after every `prune_every` inserts.
        self.policy = policy
        self.prune_every = prune_every
        self._inserts_since_prune = 0

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        self.working_memory = []

        self.reset(drop_collection=False)


    @property
    def has_camera_field(self) -> bool:
        # Collections created before multi-camera support lack the field.
        return any(f.name == 'camera_id'
                   for f in self.milv_wrapper.collection.schema.fields)

    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item)
        if not self.has_camera_field:
            memory_dict.pop('camera_id', None)
        # time alone is not collision-free at high insert rates, so add a uuid
        # suffix to keep primary keys unique and deletions precise
        memory_dict['id'] = f"{time.time()}-{uuid.uuid4().hex[:8]}"

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])

        if self.policy is not None:
            self._inserts_since_prune += 1
            if self._inserts_since_prune >= self.prune_every:
                self._inserts_since_prune = 0
                # The item above is already stored, so a failed pruning pass
                # (e.g. a transient DB timeout) must not fail the insert;
                # the next pass will pick up anything this one missed.
                try:
                    self.apply_policy(self.policy)
                except Exception as e:
                    print(f"Warning: memory pruning pass failed, will retry after {self.prune_every} more inserts: {e}")

    def get_all(self, include_embedding=True) -> List[MemoryRecord]:
        """Return every stored entry as a MemoryRecord (with absolute times)."""

        collection = self.milv_wrapper.collection
        collection.load()

        # Probe with a scalar-only query first: requesting vector output
        # fields from a completely empty collection crashes some backends
        # (milvus-lite segcore assert), and an empty scan has nothing to
        # fetch anyway. Strong consistency so unflushed inserts are seen.
        probe = collection.query(expr='id != ""', output_fields=['id'], limit=1,
                                 consistency_level='Strong')
        if not probe:
            return []

        fields = ['id', 'position', 'theta', 'time', 'caption']
        if self.has_camera_field:
            fields.append('camera_id')
        if include_embedding:
            fields.append('text_embedding')

        # Strong consistency so a pruning pass right after inserts (e.g. the
        # auto-prune inside insert()) sees every row that was just written.
        rows = []
        if hasattr(collection, 'query_iterator'):
            iterator = collection.query_iterator(batch_size=1000, expr='id != ""',
                                                 output_fields=fields, consistency_level='Strong')
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
            iterator.close()
        else:
            # Fallback for older pymilvus versions without query_iterator.
            # 16384 is the largest window a single query allows.
            rows = collection.query(expr='id != ""', output_fields=fields, limit=16384,
                                    consistency_level='Strong')

        records = []
        for row in rows:
            item = MemoryItem(
                caption=row['caption'],
                time=float(row['time'][0]) + self.time_offset,
                position=list(row['position']),
                theta=row['theta'],
                camera_id=row.get('camera_id', ''),
            )
            embedding = list(row['text_embedding']) if include_embedding else None
            records.append(MemoryRecord(id=row['id'], item=item, embedding=embedding))
        return records

    def _fetch_embedding(self, entry_id: str):
        rows = self.milv_wrapper.collection.query(
            expr=f'id == "{entry_id}"', output_fields=['text_embedding'],
            limit=1, consistency_level='Strong')
        if not rows:
            return None
        return list(rows[0]['text_embedding'])

    def remove(self, ids: list):
        """Delete entries by primary key."""

        if not ids:
            return

        collection = self.milv_wrapper.collection
        BATCH_SIZE = 256  # keep delete expressions well under Milvus expression-length limits
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i:i + BATCH_SIZE]
            expr = 'id in [' + ', '.join(f'"{entry_id}"' for entry_id in batch) + ']'
            collection.delete(expr)
        collection.flush()

    def apply_policy(self, policy: MemoryPolicy = None, now: float = None) -> PolicyResult:
        """Run a memory management policy over the collection and delete what it selects.

        Falls back to the policy configured at construction time when none is
        passed. `now` defaults to the newest stored timestamp (see
        StaleDuplicatePolicy); pass time.time() to age entries against the
        wall clock instead.
        """

        policy = policy if policy is not None else self.policy
        if policy is None:
            raise ValueError("No policy provided and no default policy configured")

        # Fetch metadata only and let the policy pull embeddings lazily:
        # it only compares stale candidates against nearby kept entries, so
        # this avoids streaming every stored 1024-d vector on each pass.
        records = self.get_all(include_embedding=False)
        for record in records:
            record.embedding_loader = lambda entry_id=record.id: self._fetch_embedding(entry_id)

        result = policy.select_for_removal(records, now=now)
        if result.drop_ids:
            self.remove(result.drop_ids)
        return result

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        if is_local_milvus(self.db_ip):
            # Reuse the Milvus Lite connection MilvusWrapper just opened:
            # langchain's Milvus cannot parse a local file uri itself, but it
            # will reuse an existing pymilvus connection with the same address.
            connection_args = connections.get_connection_addr('default')
        else:
            connection_args = {"host": self.db_ip, "port": self.db_port}

        text_vector_db = Milvus(
            self.embedder,
            connection_args=connection_args,
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args=connection_args,
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args=connection_args,
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
        )


    def search_by_position(self, query: tuple) -> str:
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        docs = similarity_search_with_score_by_vector(self.position_vector_db, np.array(query).astype(float))

        self.working_memory += docs 

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs

    def search_by_time(self, hms_time: str) -> str:

        # Input is time like 08:20:30
        # need to convert to searchable time
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
        # bad llms don't listen :(
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res: # convert to the right format then
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - self.time_offset
        # convert from hms_time to something searchable


        docs = similarity_search_with_score_by_vector(self.time_vector_db, np.array([query, 0]))

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs



    def search_by_text(self, query: str) -> str:

        docs = self.text_retriever.invoke(query)
        
        self.working_memory += docs

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs
    

    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            camera_id = doc.metadata.get('camera_id', '')
            if camera_id:
                theta = doc.metadata.get('theta')
                if theta is not None:
                    s += f"Its '{camera_id}' camera was facing {round(float(theta), 3)} radians and saw the following: {doc.page_content}\n\n"
                else:
                    s += f"Its '{camera_id}' camera saw the following: {doc.page_content}\n\n"
            else:
                s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string


# NOTE: This version of the code returns the vector
def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK.
        output_fields = pos_db.fields[:]
        # output_fields.remove(pos_db._vector_field) # NOTE: Only thing removed
        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = pos_db._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return [doc for doc, _ in ret]


