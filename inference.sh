# re10k
python -m mvsplat.src.main +experiment=re10k \
    checkpointing.load=checkpoints/mvsplat/re10k.ckpt \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=mvsplat/assets/evaluation_index_re10k_video.json \
    test.save_video=true \
    test.save_image=false \
    test.compute_scores=false \
    test.video=checkpoints/model.safetensors