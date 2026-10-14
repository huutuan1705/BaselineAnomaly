# pre-trained from VisA 
ckt_path="workspaces/models/0s-pretrained-['visa']-ViT-L-14-336-SD-VL-D4-L5-HSF-K20_best.pth"
gpu_id=0

# Industrial
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data btad
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data dagm
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data dtd
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data mpdd
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data sdd

# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data br35h
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data brain_mri
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data clinicdb
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data headct
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data isic
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data tn3k

# pre-trained from MVTec
# ckt_path="weights/pretrained_visa_clinicdb.pth"
# gpu_id=0

# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data colondb
# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --testing_model dataset --ckt_path $ckt_path --testing_data visa



