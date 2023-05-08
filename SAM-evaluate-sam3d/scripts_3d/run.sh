modality=$1
device_id=$2
include_prompt_point=$3
num_point=$4
include_prompt_box=$5
num_boxes=$6

cat scripts_3d/"$modality"_list.txt | xargs -I {} sh scripts_3d/eval.sh {} $modality $device_id $include_prompt_point $num_point $include_prompt_box $num_boxes

