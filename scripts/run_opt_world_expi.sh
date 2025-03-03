# Usage examples:

# bash scripts/camera_world/run_opt_world_chi3d.sh chi3d/train/s02 0
# bash scripts/camera_world/run_opt_world_chi3d.sh chi3d/train/s03 0
# bash scripts/camera_world/run_opt_world_chi3d.sh chi3d/train/s04 0
# bash scripts/camera_world/run_opt_world_expi.sh expi/acro1 0 expi
# bash scripts/camera_world/run_opt_world_expi.sh expi/acro2 0 expi

####################################################################

export PYTHONPATH=$PYTHONPATH:$(pwd)
cd slahmr

folder=$1
overwrite=$2
data_name=$3

# ********************************************
# Place the root of your SLAHMR directory here
root=/home/nugrinovic/code/slahmr
# ********************************************

seq_num=${1##*/}

echo "seq_num is: ${seq_num}"

search_dir=$root/videos/${folder}
echo "searching in ${search_dir}"
for entry in $search_dir/videos/*.mp4; do
  name="${entry##*/}"
  vid=${name%.*}
  video=${vid// /_}
  #  echo "Selected video is: ${name%.*}"
  echo "Selected video is: ${video}"
  if [ "$overwrite" -eq "1" ]; then
    echo "Overwrite is True, OVERWRITING!!"
    python run_opt_world.py data=${data_name} run_opt=False run_vis=True data.root=$root/videos/${data_name}/$seq_num data.seq="${video}" data.seq_id=$seq_num
  else
    if test -f "${root}/outputs/logs/${data_name}-val/${seq_num}/${video}-all-shot-0-0-180/${video}_motion_chunks_grid.mp4"; then
      if test -f "${root}/outputs/logs/${data_name}-val/${seq_num}/${video}-all-shot-0-0-180/${video}_scene_dict.pkl"; then
        echo "${video}_scene_dict.pkl exists!!! SKIPPING"
      else
        python run_opt_world.py data=${data_name} run_opt=False run_vis=True data.root=$root/videos/${data_name}/$seq_num data.seq="${video}" data.seq_id=$seq_num
      fi
    else
      echo "${video}: results not ready."
    fi

  fi

done
