#!/bin/bash

gmemFirstDimSize=(8 8 16 8 8 8)
gmemSecondDimSize=(4 4 2 4 4 4)
gmemConsecutiveRowsPaddingBytes=(0 0 512 0 0 0)
gmemAlignmentBytes=(128 32 32 128 128 128)
smemFirstDimSize=(8 8 16 8 8 8)
smemSecondDimSize=(4 4 2 4 4 4)
smemConsecutiveRowsPaddingBytes=(0 0 0 64 32 16)
smemAlignmentBytes=(128 128 128 128 128 128)
vectorSize=(4 4 4 4 4 4)

METRIC_NAME_SMEM_LDGSTS_WAVEFRONTS_COUNT="smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum "
METRIC_NAME_GMEM_LDGSTS_SECTORS_READ_COUNT="sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum "
METRIC_NAME_GMEM_LDGSTS_SECTORS_READ_L1_BYPASSED_COUNT="sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum "
METRIC_NAME_GMEM_LDGSTS_REQUESTS_COUNT="sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum "
METRIC_NAME_GMEM_LDGSTS_REQUESTS_L1_BYPASSED_COUNT="sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum "
METRIC_NAME=(
    $METRIC_NAME_SMEM_LDGSTS_WAVEFRONTS_COUNT $METRIC_NAME_GMEM_LDGSTS_SECTORS_READ_COUNT
    $METRIC_NAME_GMEM_LDGSTS_SECTORS_READ_L1_BYPASSED_COUNT $METRIC_NAME_GMEM_LDGSTS_REQUESTS_COUNT
    $METRIC_NAME_GMEM_LDGSTS_REQUESTS_L1_BYPASSED_COUNT
)
DISPLAY_NAME=(
    "#. of SMEM wavefronts"
    "#. of GMEM sectors read via L1 cache"
    "#. of GMEM sectors read bypassing L1 cache"
    "#. of GMEM LDGSTS requests via L1 cache"
    "#. of GMEM LDGSTS requests bypassing L1 cache"
)

# set -x

REPORT_PATH=./benchmark_results/async_copy_benchmark

# Run benchmark.
CONFIG_COUNT=`expr ${#gmemFirstDimSize[@]} - 1`
for iter in `seq 0 $CONFIG_COUNT`; do
    echo ""
    echo "======================================================================================"
    echo "Config params: "
    echo "GMEM buffer shape: ${gmemFirstDimSize[$iter]} x (${gmemSecondDimSize[$iter]} + ${gmemConsecutiveRowsPaddingBytes[$iter]}B)"
    echo "GMEM Alignment in Bytes: ${gmemAlignmentBytes[$iter]}"
    echo "SMEM buffer shape: ${smemFirstDimSize[$iter]} x (${smemSecondDimSize[$iter]} + ${smemConsecutiveRowsPaddingBytes[$iter]}B)"
    echo "SMEM Alignment in Bytes: ${smemAlignmentBytes[$iter]}"
    echo "#of Elements in vector: ${vectorSize[$iter]}"
    echo ""

    ncu --set full --export $REPORT_PATH --force-overwrite ./build/main 7 ${gmemFirstDimSize[$iter]} ${gmemSecondDimSize[$iter]} ${gmemConsecutiveRowsPaddingBytes[$iter]} ${gmemAlignmentBytes[$iter]} ${smemFirstDimSize[$iter]} ${smemSecondDimSize[$iter]} ${smemConsecutiveRowsPaddingBytes[$iter]} ${smemAlignmentBytes[$iter]} ${vectorSize[$iter]}
    echo ""

    METRIC_COUNT=`expr ${#METRIC_NAME[@]} - 1`
    for i in `seq 0 $METRIC_COUNT`; do
        METRIC_VALUE_LINE=$(ncu -i $REPORT_PATH.ncu-rep --page raw | grep ${METRIC_NAME[$i]} | head -1)
        # echo "METRIC_VALUE_LINE: $METRIC_VALUE_LINE"
        METRIC_VALUE_ARR=($METRIC_VALUE_LINE)
        METRIC_VALUE=${METRIC_VALUE_ARR[-1]}
        echo "${DISPLAY_NAME[$i]}: $METRIC_VALUE"
    done
done
