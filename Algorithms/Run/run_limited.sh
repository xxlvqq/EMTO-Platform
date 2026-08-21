#!/bin/bash
# <-*--*--*--*- Coder -*--*--*--*--*->
# @Introduction: 限制并行数量的运行脚本
# @Remind: 适合资源有限的服务器

# 设置最大并行任务数
MAX_PARALLEL=15

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "最大并行数: $MAX_PARALLEL"
echo "=========================================="

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "开始运行任务（每次最多 $MAX_PARALLEL 个并行）..."
echo "=========================================="

count=0
# seq 0-8: CEC17-MTSO
# seq 9-18: WCCI20-MTSO(CEC22-MTSO、WCCI22-MTSO)
# seq 19-27: C2TOP
# seq 28-33: CEC19-MaTSO
# seq 34-43: WCCI20-MaTSO
# seq 44: PEPVM
# seq 45: PKACP
# seq 46-59: MRNP
for func_num in $(seq 9 18); do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动任务 $func_num"
  python "$SCRIPT_DIR/output.py" --func "$func_num" > "$LOG_DIR/func_${func_num}.log" 2>&1 &
  
  count=$((count + 1))
  
  # 每启动 MAX_PARALLEL 个任务就等待它们完成
  if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
    echo "等待当前批次完成..."
    wait
    echo "当前批次完成，继续下一批..."
  fi
done

# 等待最后一批任务完成
wait

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务已完成！"
echo "=========================================="
