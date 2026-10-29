# clb0b5.h -> clc7b5.h

35 methods before, 43 after: 0 offsets reused by another method, 0 moved, 0 removed, 8 added, 8 with changed fields.

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

## Added

| Offset | Method |
|---|---|
| 0x021c | SET_MONITORED_FENCE_TYPE |
| 0x0220 | SET_MONITORED_FENCE_SIGNAL_ADDR_BASE_UPPER |
| 0x0224 | SET_MONITORED_FENCE_SIGNAL_ADDR_BASE_LOWER |
| 0x024c | SET_SEMAPHORE_PAYLOAD_UPPER |
| 0x0744 | SRC_ORIGIN_X |
| 0x0748 | SRC_ORIGIN_Y |
| 0x074c | DST_ORIGIN_X |
| 0x0750 | DST_ORIGIN_Y |

## Changed fields

### SET_SEMAPHORE_A (0x0240)

- `UPPER` bits 7:0 -> 16:0

### SET_SRC_PHYS_MODE (0x0260)

- added `BASIC_KIND` 5:2
- added `FLA` 9:9
- added `PEER_ID` 8:6
- `TARGET_PEERMEM`: - -> 3

### SET_DST_PHYS_MODE (0x0264)

- added `BASIC_KIND` 5:2
- added `FLA` 9:9
- added `PEER_ID` 8:6
- `TARGET_PEERMEM`: - -> 3

### LAUNCH_DMA (0x0300)

- removed `BYPASS_L2` 20:20
- added `DISABLE_PLC` 26:26
- added `FLUSH_TYPE` 25:25
- added `RESERVED_ERR_CODE` 31:28
- added `RESERVED_START_OF_COPY` 24:24
- added `SEMAPHORE_PAYLOAD_SIZE` 27:27
- `SEMAPHORE_REDUCTION_FMAX`: - -> 12
- `SEMAPHORE_REDUCTION_FMIN`: - -> 11
- `SEMAPHORE_REDUCTION_INVALIDA`: - -> 8
- `SEMAPHORE_REDUCTION_INVALIDB`: - -> 9
- `SEMAPHORE_REDUCTION_INVALIDC`: - -> 13
- `SEMAPHORE_REDUCTION_INVALIDD`: - -> 14
- `SEMAPHORE_REDUCTION_INVALIDE`: - -> 15
- `SEMAPHORE_TYPE_RELEASE_CONDITIONAL_INTR_SEMAPHORE`: - -> 3
- `SEMAPHORE_TYPE_RELEASE_SEMAPHORE_NO_TIMESTAMP`: - -> 1
- `SEMAPHORE_TYPE_RELEASE_SEMAPHORE_WITH_TIMESTAMP`: - -> 2
- added `VPRMODE` 23:22

### OFFSET_IN_UPPER (0x0400)

- `UPPER` bits 7:0 -> 16:0

### OFFSET_OUT_UPPER (0x0408)

- `UPPER` bits 7:0 -> 16:0

### SET_DST_BLOCK_SIZE (0x070c)

- `GOB_HEIGHT_GOB_HEIGHT_TESLA_4`: 0 -> -
- `WIDTH_QUARTER_GOB`: 14 -> -

### SET_SRC_BLOCK_SIZE (0x0728)

- `GOB_HEIGHT_GOB_HEIGHT_TESLA_4`: 0 -> -
- `WIDTH_QUARTER_GOB`: 14 -> -

