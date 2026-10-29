# clb06f.h -> clc56f.h

15 methods before, 27 after: 0 offsets reused by another method, 0 moved, 2 removed, 14 added, 4 with changed fields.

## Offsets now used by a different method

Writes decoded with the old layout land in the wrong state.

| Offset | Old method | New method |
|---|---|---|

## Moved

| Method | Old | New |
|---|---|---|

## Removed

| Offset | Method |
|---|---|
| 0x0024 | FB_FLUSH |
| 0x007c | CRC_CHECK |

## Added

| Offset | Method |
|---|---|
| 0x0000 | MEM_OP_C_TLB_INVALIDATE_REPLAY_NONE |
| 0x0000 | MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_NONE |
| 0x0000 | MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MIMC |
| 0x0000 | MEM_OP_D_ACCESS_COUNTER_CLR_TARGETED_TYPE_MIMC |
| 0x0004 | MEM_OP_C_TLB_INVALIDATE_REPLAY_CANCEL_GLOBAL |
| 0x0004 | MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE2 |
| 0x0028 | MEM_OP_A |
| 0x002c | MEM_OP_B |
| 0x005c | SEM_ADDR_LO |
| 0x0060 | SEM_ADDR_HI |
| 0x0064 | SEM_PAYLOAD_LO |
| 0x0068 | SEM_PAYLOAD_HI |
| 0x006c | SEM_EXECUTE |
| 0x0084 | CLEAR_FAULTED |

## Changed fields

### MEM_OP_C (0x0030)

- added `MEMBAR_TYPE` 2:0
- removed `OPERAND_LOW` 31:2
- removed `TLB_INVALIDATE_ADDR_LO` 31:12
- `TLB_INVALIDATE_PDB_ALL`: 1 -> -
- removed `TLB_INVALIDATE_TARGET` 11:10

### MEM_OP_D (0x0034)

- removed `OPERAND_HIGH` 7:0
- `OPERATION_L2_CLEAN_COMPTAGS`: 15 -> -
- `OPERATION_L2_FLUSH_DIRTY`: 16 -> -
- `OPERATION_MMU_TLB_INVALIDATE_TARGETED`: - -> 10
- removed `TLB_INVALIDATE_ADDR_HI` 7:0

### WFI (0x0078)

- `SCOPE_CURRENT_VEID`: - -> 0

### YIELD (0x0080)

- `OP_PBDMA_TIMESLICE`: 1 -> -
- `OP_RUNLIST_TIMESLICE`: 2 -> -

