# Dockerfile

ベースとなる環境

- Python3.8
- Docker (Tested on NVIDIA-Docker2)
- Webカメラ（/dev/video0に接続）
- デスクトップPC (Intel CPU + NVIDIA RTX2080Ti)

<br>

## 構築

```bash
git clone https://github.com/Kazuhito00/Image-Processing-Node-Editor.git
cd Image-Processing-Node-Editor/
docker build docker/nvidia-gpu -t ipn_editor
```

<br>

## 実行

以下のコマンドで実行できました。一度終了すると、キャッシュは全て削除されるため、任意の外部フォルダをマウントしてください。

```bash
# cd /path-to-Image-Processing-Node-Editor
xhost +
docker run --rm -it --privileged --device /dev/video0:/dev/video0:mwr -e DISPLAY=$DISPLAY -v $(pwd):/workspace --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix ipn_editor
# 勝手にウィンドウが開きます
```

### 使用したオプションについて

- `--device /dev/video0:/dev/video0:mwr`: Webカメラを接続しない場合は取り除けます。
- `--gpus all`: NVIDIA GPUでない場合は使用しないでください。
- `-v $(pwd):/workspace`: [Image-Processing-Node-Editor](https://github.com/Kazuhito00/Image-Processing-Node-Editor)のマウント先です。
  - オプションは、`Image-Processing-Node-Editor`ディレクトリ内で実行した場合です。
  - マウント先は`/workspace`となります。