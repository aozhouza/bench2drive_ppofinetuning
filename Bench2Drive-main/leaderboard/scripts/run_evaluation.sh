#!/bin/bash
# Must set CARLA_ROOT
export CARLA_ROOT=../carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=30000
export TM_PORT=50000
export DEBUG_CHALLENGE=1
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=$9
export GPU_RANK=0

# TCP evaluation
export ROUTES=leaderboard/data/bench2drive220_0_uniad_traj_20route.xml
export TEAM_AGENT=leaderboard/team_code/uniad_b2d_agent.py
export TEAM_CONFIG="/home/aozhou/bench2drive/Bench2Drive-main/Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/home/aozhou/bench2drive/Bench2Drive-main/Bench2DriveZoo/ckpts/uniad_base_b2d.pth"
# export TEAM_CONFIG="/home/aozhou/bench2drive/Bench2Drive-main/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+/home/aozhou/bench2drive/Bench2Drive-main/Bench2DriveZoo/ckpts/vad_b2d_base.pth"
export CHECKPOINT_ENDPOINT="/home/aozhou/bench2drive/Bench2Drive-main/uniad_b2d_traj/eval_bench2drive220_1231.json"
# export CHECKPOINT_ENDPOINT="/home/aozhou/bench2drive/Bench2Drive-main/vad_b2d_traj/eval_bench2drive220_0.json"
export SAVE_PATH="./eval_v1_uniad_1231/"

ITERATION=1
while true; do
    echo "===================================================="
    echo "Starting Iteration $ITERATION at $(date)"
    echo "===================================================="

    # 1. 强力清理：杀死所有残留的 CARLA 和 Python 进程
    # 防止端口 (30000, 50000) 被占用，防止显存未释放
    ps -ef | grep 'CarlaUE4' | grep -v grep | awk '{print $2}' | xargs -r kill -9
    ps -ef | grep 'leaderboard_evaluator.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
    sleep 10 # 给系统一点缓冲时间来释放句柄

    # 2. 运行 Python 脚本
    # 注意：一定要带上 --resume True
    # 设置 --epoch-limit 为 5，表示每跑 5 个 Epoch 就重启一次 CARLA
    python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
        --routes=${ROUTES} \
        --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} \
        --checkpoint=${CHECKPOINT_ENDPOINT} \
        --agent=${TEAM_AGENT} \
        --agent-config=${TEAM_CONFIG} \
        --debug=${DEBUG_CHALLENGE} \
        --record=${RECORD_PATH} \
        --resume=${RESUME} \
        --port=${PORT} \
        --traffic-manager-port=${TM_PORT} \
        --gpu-rank=${GPU_RANK} \
        --epoch-limit 10

# 获取退出状态
    EXIT_CODE=$?
    
    echo "Iteration $ITERATION finished with Exit Code: $EXIT_CODE"
    
    # 如果是因为正常的 Epoch 限制退出 (0) 或崩溃退出 (255/-1)，都继续循环
    # 只有当你手动 Ctrl+C 时，脚本才会停止（取决于你的 shell 配置）
    ITERATION=$((ITERATION+1))
    sleep 5
done
