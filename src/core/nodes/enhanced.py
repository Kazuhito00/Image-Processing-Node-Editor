#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced base node that bridges old and new architecture
This allows gradual migration of nodes to the new system
"""

import copy
import uuid
import traceback
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import dearpygui.dearpygui as dpg
else:
    try:
        import cv2
    except ImportError:
        cv2 = None

    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        dpg = None

# Import from new architecture
from src.utils.logging import get_logger
from src.utils.exceptions import NodeExecutionError, NodeConfigurationError
from src.core.nodes.base import BaseNode

logger = get_logger(__name__)


class EnhancedNode(BaseNode):
    """
    Enhanced node class that provides additional utilities
    and maintains compatibility with the old node system
    
    This class extends BaseNode with common functionality like:
    - Image conversion for DearPyGUI
    - Configuration management
    - Error handling with logging
    - Resource cleanup
    """
    
    # Constants from old system for compatibility
    TYPE_BOOLEAN = "BOOLEAN"
    TYPE_TEXT = "TEXT"
    TYPE_IMAGE = "IMAGE"
    TYPE_FLOAT = "FLOAT"
    TYPE_INT = "INT"
    TYPE_TIME_MS = "TIME_MS"
    TYPE_AUDIO = "AUDIO"
    TYPE_JSON = "JSON"
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    
    def __init__(self, node_id: int = 1, connection_dict: Optional[Dict] = None, 
                 opencv_setting_dict: Optional[Dict] = None):
        """
        Initialize enhanced node
        
        Args:
            node_id: Numeric ID for the node
            connection_dict: Connection configuration
            opencv_setting_dict: OpenCV/application settings
        """
        super().__init__()
        
        self.node_data = None
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        
        # OpenCV settings
        self._opencv_setting_dict = opencv_setting_dict if opencv_setting_dict else {}
        self.small_window_w = self._opencv_setting_dict.get('process_width', 640)
        self.small_window_h = self._opencv_setting_dict.get('process_height', 480)
        self._small_window_w = self._opencv_setting_dict.get('process_width', 640)
        self._small_window_h = self._opencv_setting_dict.get('process_height', 480)
        self.use_pref_counter = self._opencv_setting_dict.get('use_pref_counter', False)
        self.use_gpu = self._opencv_setting_dict.get('use_gpu', False)
        
        logger.debug(f"Initialized enhanced node {self.node_tag}")
    
    def convert_cv_to_dpg(self, image, width: int, height: int):
        """
        Convert OpenCV image to DearPyGUI texture format
        
        Args:
            image: OpenCV image (BGR)
            width: Target width
            height: Target height
            
        Returns:
            Texture data for DearPyGUI
        """
        if cv2 is None or np is None:
            logger.error("OpenCV or NumPy not available")
            return np.zeros(width * height * 3, dtype=np.float32) if np else None
        
        try:
            resize_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if (
                resize_image.dtype == np.uint8
                and resize_image.ndim == 3
                and resize_image.shape[2] == 3
            ):
                texture_data = cv2.cvtColor(
                    resize_image, cv2.COLOR_BGR2RGB
                ).astype(np.float32)
                texture_data /= 255.0
                return texture_data.ravel()
            data = np.flip(resize_image, 2)  # BGR to RGB
            data = np.asarray(data.ravel(), dtype=np.float32)
            texture_data = np.true_divide(data, 255.0)
            return texture_data
        except Exception as e:
            logger.error(f"Error converting image to DPG format: {e}")
            # Return empty texture on error
            return np.zeros(width * height * 3, dtype=np.float32)
    
    def get_setting_dict(self, node_id: int) -> Dict[str, Any]:
        """
        Get current settings from the GUI
        
        Args:
            node_id: Numeric ID of the node
            
        Returns:
            Dictionary of current settings
        """
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        setting_dict = {}
        
        if dpg:
            try:
                # Get position
                pos = dpg.get_item_pos(self.tag_node_name)
                setting_dict['pos'] = pos
            except Exception as e:
                logger.warning(f"Could not get position for node {node_id}: {e}")
                setting_dict['pos'] = [0, 0]
        
        setting_dict['ver'] = self._ver
        return setting_dict
    
    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]):
        """
        Apply settings to the node
        
        Args:
            node_id: Numeric ID of the node
            setting_dict: Dictionary of settings to apply
        """
        self.tag_node_name = f"{node_id}:{self.node_tag}"
        
        # Default implementation - can be overridden
        logger.debug(f"Applied settings to node {node_id}")
    
    def close(self, node_id: int):
        """
        Cleanup node resources
        
        Args:
            node_id: Numeric ID of the node
        """
        logger.debug(f"Closing node {self.node_tag} (ID: {node_id})")
        # Default implementation - can be overridden
    
    def update(
        self,
        node_id: int,
        connection_list: List[Any],
        node_image_dict: Dict[str, Any],
        node_result_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update/execute the node (default implementation)
        
        Args:
            node_id: Numeric ID of the node
            connection_list: List of connections to this node
            node_image_dict: Dictionary of images from other nodes
            node_result_dict: Dictionary of results from other nodes
            
        Returns:
            Dictionary containing the node's output
        """
        # Default implementation returns empty result
        return {"image": None, "json": None}
    
    def add_node(
        self,
        parent: Any,
        node_id: int,
        pos: List[int],
        opencv_setting_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Add node to GUI (default implementation)
        
        Args:
            parent: Parent GUI element
            node_id: Numeric ID for the node
            pos: Position [x, y] for the node
            opencv_setting_dict: Configuration dictionary
            
        Returns:
            The created node element
        """
        # Default implementation - should be overridden
        logger.warning(f"add_node not implemented for {self.node_tag}")
        return None
    
    def safe_execute(self, func: callable, *args, **kwargs) -> Any:
        """
        Execute a function with error handling and logging
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or None on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.node_tag}: {e}")
            logger.debug(traceback.format_exc())
            return None
