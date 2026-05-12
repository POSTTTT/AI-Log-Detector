"""Synthetic HDFS-format log lines for tests.

A handful of canonical HDFS event types, plus a labeled anomaly mapping,
so the pipeline can be exercised end-to-end without the real 1.5 GB dump.
"""

from __future__ import annotations

SYNTHETIC_HDFS_LOG = """\
081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_111 of size 67108864 from /10.250.19.102
081109 203616 148 INFO dfs.DataNode$PacketResponder: Received block blk_111 of size 67108864 from /10.250.19.103
081109 203617 148 INFO dfs.DataNode$PacketResponder: Received block blk_111 of size 67108864 from /10.250.19.104
081109 203620 149 INFO dfs.DataNode$DataXceiver: Receiving block blk_222 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203621 149 INFO dfs.DataNode$PacketResponder: Received block blk_222 of size 67108864 from /10.250.19.105
081109 203622 149 INFO dfs.DataNode$PacketResponder: Received block blk_222 of size 67108864 from /10.250.19.106
081109 203700 150 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/foo blk_333
081109 203701 150 INFO dfs.DataNode$DataXceiver: Receiving block blk_333 src: /10.250.19.107:54106 dest: /10.250.19.107:50010
081109 203702 150 WARN dfs.DataNode$DataXceiver: writeBlock blk_333 received exception java.io.IOException
081109 203703 150 ERROR dfs.DataNode$DataXceiver: PacketResponder blk_333 Exception java.io.IOException
081109 203800 151 INFO dfs.DataNode$PacketResponder: Received block blk_444 of size 67108864 from /10.250.19.108
"""

# block_id → 0 (normal) / 1 (anomaly)
SYNTHETIC_LABELS = {
    "blk_111": 0,
    "blk_222": 0,
    "blk_333": 1,
    "blk_444": 0,
}

SYNTHETIC_LABELS_CSV = "BlockId,Label\n" + "\n".join(
    f"{b},{'Anomaly' if v else 'Normal'}" for b, v in SYNTHETIC_LABELS.items()
)
