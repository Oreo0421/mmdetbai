#!/bin/bash
# Batch inference: Detection + Pose + Falling classification + ByteTrack
# Input:  /home/tbai/Desktop/infer/{action}/{person}/stitched_png
# Output: /home/tbai/Desktop/fall_infer/{action}/{person}

INPUT_ROOT="/home/tbai/Desktop/infer"
OUTPUT_ROOT="/home/tbai/Desktop/fall_infer"

CONFIG="/home/tbai/mmdetection/mmdetection/rtmdet_pose_v7_withclass_track.py"
CHECKPOINT="/home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_pose_v7_withclass_track/best_coco_bbox_mAP_epoch_45.pth"
SCRIPT="/home/tbai/mmdetection/mmdetection/pose_infer/infer_classifyandpred.py"

count=0
total=$(find "$INPUT_ROOT" -type d -name "stitched_png" | wc -l)

for action_dir in "$INPUT_ROOT"/*/; do
    action=$(basename "$action_dir")
    for person_dir in "$action_dir"*/; do
        person=$(basename "$person_dir")
        input_path="$person_dir/stitched_png"

        if [ ! -d "$input_path" ]; then
            echo "[SKIP] No stitched_png: $action/$person"
            continue
        fi

        count=$((count + 1))
        output_path="$OUTPUT_ROOT/$action/$person"
        mkdir -p "$output_path"

        echo ""
        echo "========================================"
        echo "[$count/$total] $action / $person"
        echo "  Input:  $input_path"
        echo "  Output: $output_path"
        echo "========================================"

        python "$SCRIPT" \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --radius 1 \
            --use-tracker \
            --input "$input_path" \
            --output "$output_path"
    done
done

echo ""
echo "========================================"
echo "All done! $count/$total folders processed."
echo "Results saved to: $OUTPUT_ROOT"
echo "========================================"
