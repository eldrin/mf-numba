import pandas as pd
from .utils import densify as _densify
from .utils import df2csr
from .validation import split_inner, split_outer, split_time, split_user

SPLIT_MAP = {
    'inner': split_inner,
    'outer': split_outer,
    'user': split_user,
    None: lambda triplet, ratio: (triplet, None)  # no split
}


class RecDataBase(object):
    """"""
    def __init__(self, data_fn, sep=',', header=None, index_col=None,
                 col_map={'user':0, 'item':1, 'value':2}, split='user',
                 split_ratio=0.8, entities=['user', 'item'], densify=None):
        """"""
        self.split = split
        self.split_ratio = split_ratio
        self.entities = entities
        self.densify = densify
        self._col_map = {v:k for k, v in col_map.items()}

        # read data
        self._triplets = RecDataBase._load_csv(
            data_fn, self._col_map, sep, header, index_col
        )
        self.prepare_data()

    @staticmethod
    def _load_csv(fn, col_map, sep=',', header=None, index_col=None):
        """"""
        data = pd.read_csv(fn, sep=sep, header=header, index_col=index_col)
        data = data[[key for key in col_map.keys() if key != 'agg']]
        if 'agg' in col_map:
            data = data.groupby([key for key in col_map.keys() if key != 'agg'])
            data = data.size().reset_index()
            data = data.rename({0:'agg'}, axis=1)
        data.columns = [col_map[col] for col in data.columns]
        return data

    def _register_internal_idx(self, triplets, entities=['user', 'item']):
        """"""
        self.entity_maps = {}
        for entity in entities:
            # 1. check and assert the un-existing entities
            if entity not in triplets.columns:
                continue

            # 2. get unique entities & register them
            self.entity_maps[entity] = {
                orig:new for new, orig
                in enumerate(set(triplets[entity].unique()))
            }

    def update_entity(self, entity, new_objects):
        """"""
        assert entity in self.entity_maps

        # update new objects
        # filter out really new objects
        last = max(self.entity_maps[entity].values()) + 1
        for o in new_objects:
            if o not in self.entity_maps[entity]:
                self.entity_maps[entity][o] = last
                last += 1

    def _prepare_mats(self, triplets):
        """"""
        # split the data
        self._train, self._test = SPLIT_MAP[self.split](
            triplets, ratio=self.split_ratio
        )

        if self.split is not None:
            # swap original object into internal indices
            for entity, entity_map in self.entity_maps.items():
                self._train[entity] = self._train[entity].map(entity_map)
                self._test[entity] = self._test[entity].map(entity_map)
            
            # convert data into CSR matrix
            mat_size = [triplets[entity].nunique() for entity in self.entities]
            self.train_mat_ = df2csr(self._train, shape=mat_size,
                                     keys=self.entities+['value'])
            self.test_mat_ = df2csr(self._test, shape=mat_size,
                                    keys=self.entities+['value'])

        else:
            # swap original object into internal indices
            for entity, entity_map in self.entity_maps.items():
                self._train[entity] = self._train[entity].map(entity_map)
            
            # convert data into CSR matrix
            mat_size = [triplets[entity].nunique() for entity in self.entities]
            self.train_mat_ = df2csr(self._train, shape=mat_size,
                                     keys=self.entities+['value'])
            self.test_mat_ = None

    def prepare_data(self):
        """"""
        # densify, if requested
        if self.densify and isinstance(self.densify, dict):
            self._triplets = _densify(self._triplets, self.densify, verbose=True)

        # register entities
        self._register_internal_idx(self._triplets, self.entities)

        # prepare train/test matrices
        self._prepare_mats(self._triplets)
