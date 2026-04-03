模型文件：
best_backbone.pth（视觉特征提取 backbone）
backbone_fusion_spec.json（输入规范、特征维度）
run_config.json（训练配置留档）
best_uar.pth（做对照组）

输入图像：RGB
尺寸：256x256
dtype：float32
值域：[0,1]（uint8/255）
不要使用 ImageNet normalize


视频转图片+清单
python -m video_to_images_for_fusion ...
输出：图片 + fusion_manifest.csv
可以视情况选择获得图片数量 我测试下来4s视频表现5帧最好
还没测试我们的视频

从图片清单提 backbone 特征
python -m csv_to_backbone_embeddings ...
输出：fusion_embeddings/*.npy + fusion_manifest_with_emb.csv

按 video_id 聚合序列特征
python -m manifest_emb_to_sequences ...
输出：fusion_sequences/*.npy（每视频一个 (T,D)）+ sequence_manifest.csv

你可以先用我跑的 sequence_manifest.csv，按 sequence_path 加载 (T,D) 特征做时序融合
标签优先用 label_id
但是后面要成为一个系统所以你也测试一下在你电脑上能不能用
