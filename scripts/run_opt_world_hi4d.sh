# Usage examples:

# bash scripts/camera_world/run_opt_world_hi4d.sh hi4d 0 hi4d 0

####################################################################


export PYTHONPATH=$PYTHONPATH:$(pwd)
cd slahmr
folder=$1
overwrite=$2
data_name=$3
in_ori=$4

# ********************************************
# Place the root of your SLAHMR directory here
root=/home/nugrinovic/code/slahmr
# ********************************************

echo "seq_num is: ${seq_num}"
echo "data_name is: ${data_name}"

search_dir=$root/videos/${folder}
echo "searching in ${search_dir}"
for entry in $search_dir/videos/*.mp4; do
  name="${entry##*/}"
  vid=${name%.*}
  video=${vid// /_}
  echo "Selected video is: ${video}"
  if [ "$overwrite" -eq "1" ]; then

    echo "Overwrite is True, OVERWRITING!!"
    python run_opt_world.py data=$data_name run_opt=False run_vis=True data.root=$root/videos/$data_name data.seq="${video}"

  else
    if test -f "${root}/outputs/logs/${data_name}-val/${video}-all-shot-0-0-180/${video}_motion_chunks_grid.mp4"; then
      if test -f "${root}/outputs/logs/${data_name}-val/${video}-all-shot-0-0-180/${video}_scene_dict.pkl"; then
        echo "${video}_scene_dict.pkl exists!!! SKIPPING"
      else
        python run_opt_world.py data=$data_name run_opt=False run_vis=True data.root=$root/videos/$data_name data.seq="${video}"
      fi
    else
      echo "${video}: results not ready."
    fi

  fi

done
