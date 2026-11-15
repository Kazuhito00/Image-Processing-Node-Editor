#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Timestamped FIFO Queue System for Node Data Communication

This module implements a timestamped queue system that allows nodes to communicate
through FIFO queues where data is timestamped and retrieved in chronological order.
"""

import time
import logging
from collections import deque
from typing import Any, Optional, Dict, Tuple
from dataclasses import dataclass
import threading

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class TimestampedData:
    """
    Container for data with timestamp information.
    
    Attributes:
        data: The actual data payload (image, audio, json, etc.)
        timestamp: Unix timestamp when the data was created
        node_id: Identifier of the node that produced this data
    """
    data: Any
    timestamp: float
    node_id: str
    
    def __lt__(self, other):
        """Compare based on timestamp for ordering"""
        return self.timestamp < other.timestamp


class TimestampedQueue:
    """
    Thread-safe buffer that stores timestamped data.
    
    Each node that sends data to other nodes uses its own buffer instance.
    Data is automatically timestamped when added to the buffer.
    The buffer maintains the most recent items up to maxsize (default 10).
    When full, oldest items are automatically removed to make room for new items.
    """
    
    def __init__(self, maxsize: int = 10, node_id: str = "unknown"):
        """
        Initialize a timestamped queue.
        
        Args:
            maxsize: Maximum number of items in the queue. When full, oldest items are removed.
            node_id: Identifier of the node owning this queue.
        """
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.RLock()
        self._node_id = node_id
        self._maxsize = maxsize
    
    def put(self, data: Any, timestamp: Optional[float] = None) -> None:
        """
        Add data to the queue with a timestamp.
        
        Args:
            data: The data to add to the queue
            timestamp: Optional custom timestamp. If None, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()
        
        timestamped_data = TimestampedData(
            data=data,
            timestamp=timestamp,
            node_id=self._node_id
        )
        
        with self._lock:
            self._queue.append(timestamped_data)
            
            # Lazy formatting: this runs for every node output on every frame
            logger.debug(
                "Queue [%s] - Inserted data: type=%s, timestamp=%.6f, queue_size=%d/%d",
                self._node_id, type(data).__name__, timestamp,
                len(self._queue), self._maxsize,
            )
    
    def get_oldest(self) -> Optional[TimestampedData]:
        """
        Retrieve the oldest data from the queue without removing it.
        
        Returns:
            The oldest TimestampedData object, or None if queue is empty.
        """
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[0]
    
    def get_latest(self) -> Optional[TimestampedData]:
        """
        Retrieve the most recent data from the queue without removing it.
        
        Returns:
            The most recent TimestampedData object, or None if queue is empty.
        """
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[-1]
    
    def pop_oldest(self) -> Optional[TimestampedData]:
        """
        Remove and return the oldest data from the queue (FIFO).
        
        Returns:
            The oldest TimestampedData object, or None if queue is empty.
        """
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue.popleft()
    
    def clear(self) -> None:
        """Remove all data from the queue."""
        with self._lock:
            self._queue.clear()
    
    def size(self) -> int:
        """Return the current number of items in the queue."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0
    
    def get_all(self) -> list:
        """
        Get all data items in the queue (oldest to newest) without removing them.
        
        Returns:
            List of TimestampedData objects ordered from oldest to newest.
        """
        with self._lock:
            return list(self._queue)


class NodeDataQueueManager:
    """
    Manages timestamped buffers for all nodes in the system.
    
    This class maintains a collection of buffers, one for each node that produces data.
    Each buffer keeps the most recent items (default 10) with timestamps for synchronization.
    It provides methods to access and manage these buffers centrally.
    """
    
    def __init__(self, default_maxsize: int = 10):
        """
        Initialize the queue manager.
        
        Args:
            default_maxsize: Default maximum size for new queues.
        """
        self._queues: Dict[str, Dict[str, TimestampedQueue]] = {}
        self._lock = threading.RLock()
        self._default_maxsize = default_maxsize
    
    def get_queue(self, node_id_name: str, data_type: str = "default") -> TimestampedQueue:
        """
        Get or create a queue for a specific node and data type.
        
        Args:
            node_id_name: The node identifier (e.g., "1:Webcam")
            data_type: Type of data (e.g., "image", "audio", "json")
        
        Returns:
            TimestampedQueue for the specified node and data type.
        """
        with self._lock:
            if node_id_name not in self._queues:
                self._queues[node_id_name] = {}
            
            if data_type not in self._queues[node_id_name]:
                self._queues[node_id_name][data_type] = TimestampedQueue(
                    maxsize=self._default_maxsize,
                    node_id=f"{node_id_name}:{data_type}"
                )
            
            return self._queues[node_id_name][data_type]

    def _find_queue(self, node_id_name: str, data_type: str) -> Optional[TimestampedQueue]:
        """Return the existing queue, or None, without creating it.

        The queue reference is resolved under the lock so a concurrent
        remove_node() cannot raise KeyError between the check and the access.
        """
        with self._lock:
            node_queues = self._queues.get(node_id_name)
            if node_queues is None:
                return None
            return node_queues.get(data_type)
    
    def put_data(self, node_id_name: str, data_type: str, data: Any, 
                 timestamp: Optional[float] = None) -> None:
        """
        Put data into a node's queue.
        
        Args:
            node_id_name: The node identifier
            data_type: Type of data
            data: The data to store
            timestamp: Optional custom timestamp
        """
        # Create timestamp once to ensure consistency across logs
        if timestamp is None:
            timestamp = time.time()
        
        queue = self.get_queue(node_id_name, data_type)
        queue.put(data, timestamp)
        
        # Log at DEBUG to avoid per-frame log pollution
        logger.debug(
            "Manager - Node [%s] received %s data at timestamp=%.6f",
            node_id_name, data_type, timestamp,
        )
    
    def get_oldest_data(self, node_id_name: str, data_type: str = "default") -> Optional[Any]:
        """
        Get the oldest data from a node's queue without removing it.
        
        Args:
            node_id_name: The node identifier
            data_type: Type of data
        
        Returns:
            The oldest data, or None if queue doesn't exist or is empty.
        """
        queue = self._find_queue(node_id_name, data_type)
        if queue is None:
            return None

        timestamped_data = queue.get_oldest()
        return timestamped_data.data if timestamped_data else None
    
    def get_latest_data(self, node_id_name: str, data_type: str = "default") -> Optional[Any]:
        """
        Get the most recent data from a node's queue without removing it.
        
        Args:
            node_id_name: The node identifier
            data_type: Type of data
        
        Returns:
            The most recent data, or None if queue doesn't exist or is empty.
        """
        queue = self._find_queue(node_id_name, data_type)
        if queue is None:
            return None

        timestamped_data = queue.get_latest()
        return timestamped_data.data if timestamped_data else None
    
    def clear_node_queues(self, node_id_name: str) -> None:
        """
        Clear all queues for a specific node.
        
        Args:
            node_id_name: The node identifier
        """
        with self._lock:
            if node_id_name in self._queues:
                for queue in self._queues[node_id_name].values():
                    queue.clear()
    
    def remove_node(self, node_id_name: str) -> None:
        """
        Remove all queues associated with a node.
        
        Args:
            node_id_name: The node identifier
        """
        with self._lock:
            if node_id_name in self._queues:
                del self._queues[node_id_name]
    
    def get_queue_info(self, node_id_name: str, data_type: str = "default") -> Dict[str, Any]:
        """
        Get information about a queue.
        
        Args:
            node_id_name: The node identifier
            data_type: Type of data
        
        Returns:
            Dictionary with queue information (size, oldest timestamp, latest timestamp, etc.)
        """
        with self._lock:
            if node_id_name not in self._queues or data_type not in self._queues[node_id_name]:
                return {
                    "exists": False,
                    "size": 0,
                }
            
            queue = self._queues[node_id_name][data_type]
            oldest = queue.get_oldest()
            latest = queue.get_latest()
            
            return {
                "exists": True,
                "size": queue.size(),
                "is_empty": queue.is_empty(),
                "oldest_timestamp": oldest.timestamp if oldest else None,
                "latest_timestamp": latest.timestamp if latest else None,
            }
