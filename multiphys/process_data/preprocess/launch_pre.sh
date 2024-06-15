#
#
#bash multiphys/preprocess/launch_pre.sh

# Declare an array of string with type
declare -a StringArray=("MPH11_00034_01"
                        "MPH112_00034_01"
                        "MPH16_00157_01"
#                        "MPH8_00034_01"
                        "N0SittingBooth_00162_01"
                        "N0Sofa_00034_01"
                   )

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
 echo $val
 python demo/preprocess/prox_to_humor_fmt.py --seq_name $val
done