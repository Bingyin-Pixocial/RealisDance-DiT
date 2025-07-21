# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref __assets__/demo/ref.png \
#     --smpl __assets__/demo/smpl.mp4 \
#     --hamer __assets__/demo/hamer.mp4 \
#     --prompt "A blonde girl is doing somersaults on the grass. Behind the grass is a river, \
#     and behind the river are trees and mountains. The girl is wearing black yoga pants and a black sports vest." \
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/1.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
#                 She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output_corrected_inference \
#     --multi-gpu



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/1.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
#                 She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
#     --num-frames 493 \
#     --fps 16 \
#     --save-dir ./output_corrected_inference_16fps \
#     --multi-gpu

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_100step.py \
#     --ref pose/ref/1.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
#                 She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output_40step_neg_prompt \
#     --multi-gpu


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/1.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
#                 She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
#     --num-frames 151 \
#     --fps 30 \
#     --save-dir ./output_corrected_inference_30fps \
#     --multi-gpu



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --ref pose/ref/1.png \
    --smpl pose/smpl/1.mp4 \
    --hamer pose/hamer/1.mp4 \
    --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
                She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
    --num-frames 151 \
    --fps 30 \
    --save-dir ./output_30fps \
    --multi-gpu


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --ref pose/ref/2.png \
    --smpl pose/smpl/1.mp4 \
    --hamer pose/hamer/1.mp4 \
    --prompt "A cheerful young Black man is dancing in front of a white board. \
                He’s wearing a well-fitted blue short-sleeve T-shirt, slim black jeans, and black lace-up high-top sneakers."\
    --num-frames 151 \
    --fps 30 \
    --save-dir ./output_30fps \
    --multi-gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --ref __assets__/demo/ref.png \
    --smpl pose/smpl/1.mp4 \
    --hamer pose/hamer/1.mp4 \
    --prompt "A blonde girl is dancing on the grass. Behind the grass is a river, \
    and behind the river are trees and mountains. The girl is wearing black yoga pants and a black sports vest." \
    --num-frames 151 \
    --fps 30 \
    --save-dir ./output_30fps \
    --multi-gpu

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --ref pose/ref/3.jpg \
    --smpl pose/smpl/1.mp4 \
    --hamer pose/hamer/1.mp4 \
    --prompt "A girl is dancing indoors against a grey and wood-paneled wall. \
        The girl is wearing a cropped white tank top, plaid jacket, white pleated skirt, and nude tights, \
        with a brown shoulder bag."\
    --num-frames 151 \
    --fps 30 \
    --save-dir ./output_30fps \
    --multi-gpu


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
    --ref pose/ref/4.jpg \
    --smpl pose/smpl/1.mp4 \
    --hamer pose/hamer/1.mp4 \
    --prompt "A girl is dancing on a sunlit street in front of a brick building. \
        The girl is wearing a red gingham mini dress, white sneakers with Nike socks, and a black shoulder bag."\
    --num-frames 151 \
    --fps 30 \
    --save-dir ./output_30fps \
    --multi-gpu



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref __assets__/demo/ref.png \
#     --smpl __assets__/demo/smpl.mp4 \
#     --hamer __assets__/demo/hamer.mp4 \
#     --prompt "A blonde girl is doing somersaults on the grass. Behind the grass is a river, \
#     and behind the river are trees and mountains. The girl is wearing black yoga pants and a black sports vest." \
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/1.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A smiling young blonde woman with long hair is dancing in front of a white board. \
#                 She’s a snug white tank top tucked into slim-fit blue jeans and soft blue ankle socks."\
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/2.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A cheerful young Black man is dancing in front of a white board. \
#                 He’s wearing a well-fitted blue short-sleeve T-shirt, slim black jeans, and black lace-up high-top sneakers."\
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/3.jpg \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A girl is dancing indoors against a grey and wood-paneled wall. \
#         The girl is wearing a cropped white tank top, plaid jacket, white pleated skirt, and nude tights, \
#         with a brown shoulder bag."\
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference.py \
#     --ref pose/ref/4.jpg \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A girl is dancing on a sunlit street in front of a brick building. \
#         The girl is wearing a red gingham mini dress, white sneakers with Nike socks, and a black shoulder bag."\
#     --num-frames 81 \
#     --fps 16 \
#     --save-dir ./output_5s \
#     --multi-gpu





# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_100step.py \
#     --ref pose/ref/2.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A cheerful young Black man is dancing in front of a white board. \
#                 He’s wearing a well-fitted blue short-sleeve T-shirt, slim black jeans, and black lace-up high-top sneakers."\
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output \
#     --multi-gpu

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_100step.py \
#     --ref __assets__/demo/ref.png \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A blonde girl is dancing on the grass. Behind the grass is a river, \
#     and behind the river are trees and mountains. The girl is wearing black yoga pants and a black sports vest." \
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output \
#     --multi-gpu

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_100step.py \
#     --ref pose/ref/3.jpg \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A girl is dancing indoors against a grey and wood-paneled wall. \
#         The girl is wearing a cropped white tank top, plaid jacket, white pleated skirt, and nude tights, \
#         with a brown shoulder bag."\
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output \
#     --multi-gpu


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 inference_100step.py \
#     --ref pose/ref/4.jpg \
#     --smpl pose/smpl/1.mp4 \
#     --hamer pose/hamer/1.mp4 \
#     --prompt "A girl is dancing on a sunlit street in front of a brick building. \
#         The girl is wearing a red gingham mini dress, white sneakers with Nike socks, and a black shoulder bag."\
#     --num-frames 493 \
#     --fps 29 \
#     --save-dir ./output \
#     --multi-gpu

