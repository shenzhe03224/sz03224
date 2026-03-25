<<<<<<< HEAD
# 工业轮子三维重建与验证

本项目用于搭建“工业轮子三维重建与验证”流程的最小可运行骨架，当前阶段只提供目录约定、脚本入口、参数解析与基础输入输出检查，不实现复杂算法。

后续可以在这个骨架上逐步补充以下能力：

- 图像预处理与筛选
- 轮子区域分割
- 基于 COLMAP 的三维重建
- 与标准 `XYZ` 点云的对比验证

## 目录结构

```text
wheel_recon_project/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- images_2d/          # 原始二维图像
|   |-- gt_xyz/             # 标准 XYZ 点云
|   `-- depth_meta/         # 深度相关元数据（可选）
|-- outputs/
|   |-- prepared_images/    # 图像准备阶段输出
|   |-- segmentation/       # 分割阶段输出
|   |-- colmap_workspace/   # COLMAP 工作目录
|   `-- evaluation/         # XYZ 验证结果
`-- scripts/
    |-- utils.py
    |-- prepare_images.py
    |-- segment_wheel.py
    |-- run_colmap.py
    `-- evaluate_xyz.py
```

## 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

## 运行顺序

当前推荐按下面顺序执行，每一步都会打印输入输出路径，并做基础检查。

### 1. 图像准备

扫描 `data/images_2d`，读取图像并生成基础质量报告。当前会统计图像尺寸、灰度均值和拉普拉斯方差，并按阈值做基础筛选。

```bash
python scripts/prepare_images.py --input-dir data/images_2d --output-dir outputs/prepared_images --blur-threshold 100 --brightness-threshold 40 --copy-selected false
```

输出：

- `outputs/prepared_images/image_report.csv`
- `outputs/prepared_images/images_selected/`
- `outputs/prepared_images/images_rejected/`

### 2. 轮子分割任务准备

读取准备好的轮子图像，使用传统图像处理方法生成二值 mask、叠加预览图和分割报告。

```bash
python scripts/segment_wheel.py --input-dir outputs/prepared_images_v2/images_selected --mask-dir data/masks --preview-dir outputs/segmentation/mask_preview --report-path outputs/segmentation/mask_report.csv --min-area-ratio 0.03 --debug false
```

输出：

- `data/masks/`
- `outputs/segmentation/mask_preview/`
- `outputs/segmentation/mask_report.csv`

### 3. COLMAP 重建命令准备

检查图像目录，创建 COLMAP 工作区，并生成待执行命令列表。默认只写命令；传入 `--run` 时会按顺序真实执行 `feature_extractor`、matcher 和 `mapper`。

```bash
python scripts/run_colmap.py --colmap-path colmap --image-dir outputs/prepared_images_v2/images_selected --mask-dir data/masks --workspace outputs/colmap_workspace --database-path outputs/colmap_workspace/database.db --camera-model SIMPLE_RADIAL --matcher exhaustive --use-masks false
```

输出：

- `outputs/colmap_workspace/run_colmap_commands.txt`

### 3.5 fused 点云清理

对 `outputs/colmap_workspace/dense/fused.ply` 做后处理，输出下采样版、去噪版和主簇版点云，便于得到更干净的轮子主体点云。

```bash
python scripts/clean_fused_pointcloud.py --input-ply outputs/colmap_workspace/dense/fused.ply --output-dir outputs/cleaned_pointcloud --voxel-size 0.005 --nb-neighbors 20 --std-ratio 2.0 --dbscan-eps 0.03 --dbscan-min-points 30
```

输出：

- `outputs/cleaned_pointcloud/fused_downsampled.ply`
- `outputs/cleaned_pointcloud/fused_denoised.ply`
- `outputs/cleaned_pointcloud/fused_main_cluster.ply`
- `outputs/cleaned_pointcloud/clean_report.txt`

### 3.6 轮子网格重建

对清理后的主簇点云执行法向估计、Poisson 表面重建、低密度顶点剔除和简单平滑，得到更连续、更适合展示的轮子表面网格。当前脚本的目标是提升展示连续性，不代表真实精度一定提高。

```bash
python scripts/reconstruct_wheel_mesh.py --input-ply outputs/cleaned_pointcloud/fused_main_cluster.ply --output-dir outputs/wheel_mesh --normal-radius 0.03 --max-nn 30 --poisson-depth 8 --density-quantile 0.02 --smooth-iterations 5
```

输出：

- `outputs/wheel_mesh/wheel_mesh_poisson.ply`
- `outputs/wheel_mesh/wheel_mesh_smooth.ply`
- `outputs/wheel_mesh/wheel_mesh.obj`
- `outputs/wheel_mesh/mesh_report.txt`

### 4. XYZ 基础验证

读取预测点云与标准点云，输出基础统计信息，并基于点数最多的 GT 先做尺度对齐和质心粗对齐，再做一次基础 ICP 配准验证。当前版本仅用于基础验证，不代表工业级精度评估；如果 GT 是单视角局部点云，结果仅供参考。

```bash
python scripts/evaluate_xyz.py --pred-xyz outputs/colmap_workspace/dense/fused.ply --gt-dir data/gt_xyz --output-dir outputs/evaluation
```

输出：

- `outputs/evaluation/xyz_summary.csv`
- `outputs/evaluation/icp_result.txt`
- `outputs/evaluation/pred_scaled_aligned_pre_icp.ply`
- `outputs/evaluation/pred_downsampled.ply`
- `outputs/evaluation/gt_downsampled.ply`
- `outputs/evaluation/pred_registered.ply`
- `outputs/evaluation/distance_hist.png`

## 脚本说明

### `scripts/utils.py`

通用工具函数，包含：

- 路径转换与目录创建
- 输入目录与文件检查
- 图像与 `XYZ` 文件扫描
- 终端信息打印

### `scripts/prepare_images.py`

图像准备阶段骨架：

- 解析输入输出参数
- 扫描并排序图像目录
- 检查图像是否可读取
- 统计宽、高、亮度均值与清晰度
- 生成筛选报告 CSV
- 可选复制通过筛选和 rejected 的图像

### `scripts/segment_wheel.py`

分割阶段骨架：

- 解析输入输出参数
- 读取图像并进行灰度、CLAHE、平滑和 Otsu 分割
- 使用形态学与轮廓筛选轮子区域
- 输出 mask、预览图和分割报告

### `scripts/run_colmap.py`

重建阶段骨架：

- 检查 `COLMAP` 可执行程序
- 创建数据库和稀疏重建工作目录
- 生成或执行 `feature_extractor`、matcher 和 `mapper`

### `scripts/evaluate_xyz.py`

验证阶段骨架：

- 检查预测点云与标准点云路径
- 读取 `PLY` 或 `XYZ` 点云
- 输出点数、包围盒和质心统计
- 自动选择 GT，执行尺度对齐、质心粗对齐和基础 ICP 验证
- 导出配准结果和距离直方图

### `scripts/clean_fused_pointcloud.py`

重建后处理脚本：

- 读取 `fused.ply`
- 执行下采样、统计去噪和可选半径去噪
- 使用 DBSCAN 提取最大主簇
- 导出清理结果和清理报告

### `scripts/reconstruct_wheel_mesh.py`

网格重建脚本：

- 读取清理后的主簇点云
- 估计并一致化法向
- 执行 Poisson 表面重建
- 剔除低密度顶点并做简单平滑
- 导出展示用网格和重建报告

## 当前阶段边界

为了保持骨架最小、可运行、便于后续扩展，当前版本刻意不包含以下内容：

- 图像增强、去畸变、裁剪等复杂预处理
- 真实轮子分割模型或传统视觉分割算法
- 自动执行 COLMAP 重建
- ICP、Chamfer Distance、配准误差等复杂评估指标

后续如果你愿意，我可以继续在这个骨架上帮你补第二步，把每个脚本逐步实现成真正可跑的处理流程。
=======
# sz03224
>>>>>>> 4b429a26f9a5d61314ed42fc4b40961b77634309
