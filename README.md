[Japanese/[English](README_EN.md)]

# Image-Processing-Node-Editor
ノードエディターベースの画像処理アプリです。<br>
処理の検証や比較検討での用途を想定しています。<br>

<img src="https://user-images.githubusercontent.com/37477845/172011014-23fb025e-68a5-4cb7-925f-c4417029966c.gif" loading="lazy" width="100%">

# Note
ノードは作成者(高橋)が必要になった順に追加しているため、<br>
画像処理における基本的な処理を担うノードが不足していることがあります。<br>

# Requirement
```
opencv-python   4.5.5.64 or later
onnxruntime-gpu 1.12.0   or later
dearpygui       1.6.2    or later
mediapipe       0.8.10   or later ※mediapipeノード実行に必要
protobuf        3.20.0   or later ※mediapipeノード実行に必要
filterpy        1.4.5    or later ※motpyノード実行に必要
lap             0.4.0    or later ※ByteTrackノード実行に必要
Cython          0.29.30  or later ※ByteTrackノード実行に必要
cython-bbox     0.1.3    or later ※ByteTrackノード実行に必要
rich            12.4.4   or later ※Norfairノード実行に必要
```

※Windowsでcython_bbox のインストールが失敗する場合は、numpy、Cythonをインストールしてから<br>　cython-bboxはGitHubからのインストールをお試しください(2022/06/05時点)<br>

```
pip install numpy
pip install Cython
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

# Installation
以下の何れかの方法で環境を準備してください。<br>
* スクリプトを直接実行
    1. リポジトリをクローン<br>`git clone https://github.com/Kazuhito00/Image-Processing-Node-Editor`
    1. パッケージをインストール <br>`pip install -r requirements.txt`  
    1. 「main.py」を実行<br>`python main.py`
* Dockerを利用
    1. [Image-Processing-Node-Editor/docker/nvidia-gpu](https://github.com/Kazuhito00/Image-Processing-Node-Editor/tree/main/docker/nvidia-gpu) を参照
* 実行ファイルを利用(Windowsのみ)
    1. [ipn-editor_win_x86_64.zip](https://github.com/Kazuhito00/Image-Processing-Node-Editor/releases/download/v0.1.1/ipn-editor_win_x86_64.zip)をダウンロード
    1. 「main.exe」を実行 
* pipインストールを利用<br><b>※インストールされるディレクトリ名が「node」「node_editor」となってしまうため修正予定<br>→pip利用時はvenv等の仮想環境でのインストールを強く推奨 </b>
    1. ビルドツールをインストール<br>Windows：https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/<br>Ubuntu：`sudo apt-get install build-essential libssl-dev libffi-dev python3-dev`
    1. Numpy、Cython、wheelをインストール<Br>`pip install Cython numpy wheel`
    1. GitHubリポジトリを指定し、pipインストール<br>`pip install git+https://github.com/Kazuhito00/Image-Processing-Node-Editor`
    1. 以下コマンドでアプリケーションを起動<br>`ipn-editor`  

# Usage
アプリの起動方法は以下です。
```bash
python main.py
```
* --setting<br>
ノードサイズやVideoWriterの設定が記載された設定ファイルパスの指定<br>
デフォルト：node_editor/setting/setting.json
* --unuse_async_draw<br>
非同期描画を使用しない<Br>→GUIイベントループとノードの更新処理を直列に実施<br>※ノード異常終了時などの原因調査用<br>
デフォルト：指定なし

### Create Node
メニューから作成したいノードを選びクリック<br>
<img src="https://user-images.githubusercontent.com/37477845/172030402-80d3d14e-d0c8-464f-bb0c-139bfe676845.gif" loading="lazy" width="50%">

### Connect Node
出力端子をドラッグして入力端子に接続<br>
端子に設定された型同士のみ接続可能<br>
<img src="https://user-images.githubusercontent.com/37477845/172030403-ec4f0a89-22d5-4467-9b11-c8e595e65997.gif" loading="lazy" width="50%">

### Delete Node
削除したいノードを選択した状態で「Del」キー<br>
<img src="https://user-images.githubusercontent.com/37477845/172030418-201d7df5-1984-4fa7-8e47-9264c5dcb6cf.gif" loading="lazy" width="50%">

### Export
メニューから「Export」を押し、ノード設定(jsonファイル)を保存<br>
<img src="https://user-images.githubusercontent.com/37477845/172030429-9c6c453c-b8b0-4ccf-b36e-eb666c2d919f.gif" loading="lazy" width="50%">

### Import
Exportで出力したノード設定(jsonファイル)を読み込む<br>
<img src="https://user-images.githubusercontent.com/37477845/172030433-8a07b702-9ba4-43e7-9f2f-f0885f472c44.gif" loading="lazy" width="50%">

# Node
<details>
<summary>Input Node</summary>

<table>
    <tr>
        <td width="200">
            Image
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031017-fd0107a5-2a33-4e47-a18b-ea53213f65e1.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            静止画(bmp, jpg, png, gif)を読み込み、画像を出力するノード<br>
            「Select Image」ボタンでファイルダイアログをオープン
        </td>
    </tr>
    <tr>
        <td width="200">
            Video
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031118-9382a9f6-d45c-4d39-ae82-59575a109664.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            動画(mp4, avi)を読み込み、フレーム毎の画像を出力するノード<br>
            「Select Movie」ボタンでファイルダイアログをオープン<br>
            動画をループ再生する場合は「Loop」にチェック<br>
            「Skip Rate」は動画読み込み時に、何フレームに1回出力するか指定する数値
        </td>
    </tr>
    <tr>
        <td width="200">
            WebCam
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031202-2ec0e976-12c7-41a9-94e4-ef162302f0b1.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            Webカメラを読み込み、フレーム毎の画像を出力するノード<br>
            「Device No」ドロップダウンリストでカメラ番号を指定<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            RTSP
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/178135453-293836c2-e38d-476f-9b64-ea654470ba2e.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            ネットワークカメラのRTSP入力を読み込み、フレーム毎の画像を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Int Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031284-95255053-6eaf-4298-a392-062129e698f6.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            整数値を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Float Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031323-98ae0273-7083-48d0-9ef2-f02af7fde482.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            フロート値を出力するノード<br>
        </td>
    </tr>
</table>
</details>

<details>
<summary>Process Node</summary>

<table>
    <tr>
        <td width="200">
            ApplyColorMap
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031657-81e70c61-05a3-4bff-9423-67ac9e486f5c.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に疑似カラーを適用し、疑似カラー画像を出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Blur
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031667-399472c9-7731-4cc2-8258-6879a1836b66.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し平滑化処理を実行し、平滑化画像を出力するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Brightness
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031761-9ab8d83d-9bac-4854-9a6d-44c34692a002.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し輝度調整処理を実行し、輝度調整画像を出力するノード<br>
            「alpha」スライドバーで輝度調整値を変更可能<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Canny
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172032723-df30d0bb-ed24-4909-afee-c3a78f66dad9.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しキャニー法を用いたエッジ検出処理を実行し<br>エッジ検出画像を出力するノード<br>
            スライダーで最小閾値と最大閾値を指定
        </td>
    </tr>
    <tr>
        <td width="200">
            Contrast
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042432-dab55644-f95f-4854-bcc4-45bb54d9c5bd.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しコントラスト調整処理を実行し、コントラス調整画像を出力するノード<br>
            「beta」スライドバーでコントラスト調整値を変更可能<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Crop
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042627-1c90f1ca-2d57-45b4-8dbe-ce0e4917d08e.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像の切り抜きを実行し、切り抜き画像を出力するノード<br>
            左上座標(x1, y1)と右上座標(x2, y2)をスライダーで変更可能<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            EqualizeHist
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042718-4f14021f-c29e-4886-b44f-46af644a74fe.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像の明度部分のヒストグラム平坦化を実行し、画像を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Flip
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042828-62d5ba24-69f9-4d6b-b3f9-322f43af0284.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し水平反転/垂直反転を実行し、画像を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Gamma Correction
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042880-7804d210-72f7-4977-ac11-41f9e7883a65.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しガンマ補正を実行し、画像を出力するノード<br>
            スライダーでγ値を変更可能
        </td>
    </tr>
    <tr>
        <td width="200">
            Grayscale
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042929-1501d980-b00b-42f7-bbb3-a078d95be5ff.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像をグレースケール化し、画像を出力するノード<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Threshold
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172042985-3e7908cc-f485-4684-884c-8cfe3d020004.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像を2値化し、画像を出力するノード<br>
            「type」で2値化アルゴリズムを指定<br>
            「threshold」で閾値変更<br><br>
            「type」で「大津の2値化(THRESH_OTSU)」は<br>
            閾値自動決定アルゴリズムのため「threshold」値は無視
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple Filter
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/178098739-ee15159c-d66f-4b5d-822d-dbaf686448d6.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に3×3の2次元フィルタリング処理を行い、画像を出力するノード
        </td>
    </tr>
</table>
</details>


<details>
<summary>Deep Learning Node</summary>

ドロップダウンリストでモデルを指定し、CPU/GPUチェックボックスで推論時のデバイスを変更可能<br>
※モデルがGPU推論に対応していない場合はGPUにチェックを入れてもCPU推論<br>
ノードが使用するモデルのライセンスは「node/deep_learning_node/XXXXXXXX/」の各ディレクトリを参照
<table>
    <tr>
        <td width="200">
            Classification
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172043243-2c037f0b-e1ba-4e3b-96a8-b0e3358f6616.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しクラス分類を実行するノード<br>
            出力画像は未加工の画像<br><br>
            Object Detectionノードを接続した場合<br>バウンディングボックスに対しクラス分類を実行
        </td>
    </tr>
    <tr>
        <td width="200">
            Face Detection
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172045704-23c00432-90b1-4a53-b621-6413ba8f18dd.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し顔検出を実行するノード<br>
            出力画像は未加工の画像
        </td>
    </tr>
    <tr>
        <td width="200">
            Low-Light Image Enhancement
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172045825-8ad902e0-d11d-44b7-8390-bb3e7ab12622.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し暗所ノイズ除去(Low-Light Image Enhancement)を実行するノード<br>
            出力画像はノイズ除去適用済みの画像
        </td>
    </tr>
    <tr>
        <td width="200">
            Monocular Depth Estimation
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172045864-8e249b46-d5bf-4d48-b540-2e5102afbe21.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し単眼深度推定を実行するノード<br>
            出力画像は単眼深度推定を適用しグレースケール化した画像
        </td>
    </tr>
    <tr>
        <td width="200">
            Object Detection
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172044154-1ef0a081-0e1e-4e3f-8d0d-599b73ee895d.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し物体検出を実行するノード<br>
            出力画像は未加工の画像
        </td>
    </tr>
    <tr>
        <td width="200">
            Pose Estimation
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172045920-cf18889d-d2f8-43ba-b3a5-773fd8df7eec.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対し姿勢推定を実行するノード<br>
            出力画像は未加工の画像
        </td>
    </tr>
    <tr>
        <td width="200">
            Semantic Segmentation
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172045965-6d77f4ef-d208-40c9-a335-25a9d1d07acc.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しセマンティックセグメンテーションを実行するノード<br>
            出力画像は未加工の画像
        </td>
    </tr>
    <tr>
        <td width="200">
            QR Code Detection
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/174199447-f92a18ef-cc76-46a3-abf5-314f8f9e01fe.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像に対しQRコード検出を実行するノード<br>
            出力画像は未加工の画像
        </td>
    </tr>
</table>
</details>

<details>
<summary>Analysis Node</summary>

<table>
    <tr>
        <td width="200">
            FPS
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172046425-ad00b7ea-b91b-4542-81d2-c92002f8a925.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            ノードの処理時間(ms)を元にFPSを算出するノード<br>
            「Add Slot」で処理時間入力端子を追加可能
        </td>
    </tr>
    <tr>
        <td width="200">
            RGB Histgram
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172046609-45ce392e-cbf1-4f14-b4eb-ee6b3fe7cc80.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像のRGB各チャンネルのヒストグラムを算出して<br>
            グラフに表示するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            BRISQUE
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/173472170-cc47e04e-80e7-4126-949f-a0f034b9f0b8.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            BRISQUEを用いた画質評価を行うノード<br>
            ※数値が高いほど悪い
        </td>
    </tr>
</table>
</details>

<details>
<summary>Draw Node</summary>

<table>
    <tr>
        <td width="200">
            Draw Information
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172046789-0d43ca22-b202-404a-ba01-dd80a01d01e5.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            ClassificationノードやObject Detectionノードなどの<br>
            未加工画像を出力するノードの画像に対して、<br>
            解析結果を描画する
        </td>
    </tr>
    <tr>
        <td width="200">
            Image Concat
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172046873-1bb27261-160a-452e-b454-05d249ec1aca.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            複数入力画像を並べて表示するノード<br>
            「Add Slot」で画像入力端子を追加可能
        </td>
    </tr>
    <tr>
        <td width="200">
            PutText
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172046942-7d004807-348d-4576-bac5-f4da27f0e5ed.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像の左上にテキストを描画するノード<br>
            描画色はカラーマップで選択可能<br>
            処理時間入力端子を接続することで処理時間もあわせて描画
        </td>
    </tr>
    <tr>
        <td width="200">
            Result Image
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172047088-eb867eab-98bf-4f46-8435-533f03a8f9b0.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            画像を表示するノード<br>
            処理ノードよりも大きい表示を行う<br>
            また、ClassificationノードやObject Detectionノードなどの<Br>
            未加工画像を出力するノードを接続すると解析結果を追加して描画
        </td>
    </tr>
    <tr>
        <td width="200">
            Result Image(Large)
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172047088-eb867eab-98bf-4f46-8435-533f03a8f9b0.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            Result Imageノードよりも大きく表示
        </td>
    </tr>
</table>
</details>

<details>
<summary>Other Node</summary>

<table>
    <tr>
        <td width="200">
            ON/OFF Switch
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172047545-e0887c75-16d0-450e-8cc2-50f4065173e0.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像を出力するか切り替えるノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Video Writer
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172047578-7ee450ff-0816-4006-814f-55f854ca921a.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            入力画像を動画をして書き出すノード<br>
            出力先、出力サイズ、FPSは「setting.json」にて指定
        </td>
    </tr>
</table>
</details>

<details>
<summary>Preview Release Node</summary>

今後大きく仕様を変更する可能性のあるノード
<table>
    <tr>
        <td width="200">
            MOT
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172049681-67df2cc3-3db3-4766-a96e-f7c557e4a5b9.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            Object Detectionノードを入力しMOT(Multi Object Tracking)を実行するノード
        </td>
    </tr>
    <tr>
        <td width="200">
            Exec Python Code
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/179454389-7b707584-ef3b-43f2-8e99-db74005c76e8.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            Pythonコードを実行するノード<br>
            入力画像用の変数は「input_image」<br>
            出力画像用の変数は「output_image」
        </td>
    </tr>
</table>
</details>

# Node(Another repository)
他リポジトリで公開しているノードです。<br>
Image-Processing-Node-Editor で使用するには、各リポジトリのインストール方法に従ってください。

<details>
<summary>Input Node</summary>

<table>
    <tr>
        <td width="200">
            YouTube
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/179450682-f7cc8237-e9d8-4c0f-b5d8-d2caac453f04.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            YouTubeを読み込み、画像を出力するノード<br>
            URL欄にYouTube動画のURLを指定して「Start」ボタンを押してください<br>
            再生が始まるまでに少々時間がかかります<br>
            Interval(ms)でYouTube読み込み間隔を指定します
        </td>
    </tr>
</table>

</details>

# ToDo
- [ ] RGB Histgramノードのグラフ部分が常に最前面に表示される問題の調査
- [ ] 複数ノードを接続したノードを削除した際に接続線が残る問題の調査
- [ ] Import機能がノード追加前にしか利用できない挙動の修正
  
# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Image-Processing-Node-Editor is under [Apache-2.0 license](LICENSE).<br><br>
Image-Processing-Node-Editorのソースコード自体は[Apache-2.0 license](LICENSE)ですが、<br>
各アルゴリズムのソースコードは、それぞれのライセンスに従います。<br>
詳細は各ディレクトリ同梱のLICENSEファイルをご確認ください。

# License(Image)
サンプルで表示している画像は[フリー素材ぱくたそ](https://www.pakutaso.com/)様、[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)様からお借りしています。
