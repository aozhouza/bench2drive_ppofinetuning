#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True

# === 修改点 1: 直接指定你的 XML 文件路径 ===
# 请确保这个路径是真实存在的
SPECIFIC_ROUTES="leaderboard/data/bench2drive220_0_uniad_traj_oneroute.xml"

TEAM_AGENT=team_code/uniad_b2d_agent.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+YOUR_CKPT_PATH/uniad_base_b2d.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=uniad
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# === 修改点 2: 移除了 split_xml 的逻辑 ===
echo -e "\033[32m Skipping split_xml.py (Single GPU / Custom Route Mode) \033[0m"

echo -e "**************\033[36m Single GPU Configuration \033[0m **************"
# === 修改点 3: 只配置一张卡 ===
GPU_RANK_LIST=(0)
TASK_LIST=(0)
echo -e "\033[32m GPU_RANK_LIST: ${GPU_RANK_LIST[*]} \033[0m"
echo -e "\033[32m TASK_LIST: ${TASK_LIST[*]} \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    
    # === 修改点 4: 使用指定的 XML 文件 ===
    ROUTES=$SPECIFIC_ROUTES
    
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m ROUTES XML: $ROUTES \033[0m"
    
    echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    
    # 修改了 Log 文件名，避免原逻辑的变量缺失导致名字奇怪
    LOG_FILE="single_gpu_run.log"
    
    bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > $LOG_FILE &
    sleep 5
done
wait