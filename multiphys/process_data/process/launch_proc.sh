
# to13
# bash multiphys/process/launch_proc.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/nugrinovic/.mujoco/mujoco210/bin
# Declare an array of string with type
declare -a StringArray=(
                        "MPH11_00034_01"
                        "MPH112_00034_01"
                        "MPH16_00157_01"
#                        "MPH8_00034_01"
#                        "N0SittingBooth_00162_01"
#                        "N0Sofa_00034_01"
                   )

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
 echo $val
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 # --to_floor 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --noisy_rot 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --noisy_rot 1 --noisy_rotx 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --noisy_rotz 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --set_conf_ones 1
# python multiphys/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --noisy_2d 10
 python demo/process/process_prox.py --use_dekr 1 --seq_name $val --ignore_align 0 --to_floor 1 --noisy_2d 100
done

#python multiphys/process/process_prox.py --use_dekr 1 --seq_name MPH1Library_00034_01


