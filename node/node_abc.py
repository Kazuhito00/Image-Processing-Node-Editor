from abc import ABCMeta, abstractmethod


class DpgNodeABC(metaclass=ABCMeta):
    _ver = '0.0.0'

    node_label = ''
    node_tag = ''

    TYPE_INT = 'Int'
    TYPE_FLOAT = 'Float'
    TYPE_IMAGE = 'Image'
    TYPE_TIME_MS = 'TimeMS'
    TYPE_TEXT = 'Text'

    @abstractmethod
    def add_node(
        self,
        parent,
        node_id,
        pos,
        width,
        height,
        opencv_setting_dict,
    ):
        pass

    @abstractmethod
    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        pass

    @abstractmethod
    def get_setting_dict(self, node_id):
        pass

    @abstractmethod
    def set_setting_dict(self, node_id, setting_dict):
        pass

    @abstractmethod
    def close(self, node_id):
        pass
