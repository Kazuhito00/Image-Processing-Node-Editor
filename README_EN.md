[[Japanese](https://github.com/Kazuhito00/Image-Processing-Node-Editor)/English] 

# Image-Processing-Node-Editor
An application that performs image processing with the node editor.<br>
It is used for processing verification and comparison.<br>

<img src="https://user-images.githubusercontent.com/37477845/172011014-23fb025e-68a5-4cb7-925f-c4417029966c.gif" loading="lazy" width="100%">

# Note
Since the nodes are added in the order in which the author(Takahashi) needs them,<br>
There may be a shortage of nodes responsible for basic processing in image processing.<br>

# Requirement
```
opencv-python   4.5.5.64 or later
onnxruntime-gpu 1.11.1   or later
dearpygui       1.6.2    or later
mediapipe       0.8.10   or later ※Required to run mediapipe node
protobuf        3.20.0   or later ※Required to run mediapipe node
filterpy        1.4.5    or later ※Required to run MOT(motpy) node
lap             0.4.0    or later ※Required to run MOT(ByteTrack) node
Cython          0.29.30  or later ※Required to run MOT(ByteTrack) node
cython-bbox     0.1.3    or later ※Required to run MOT(ByteTrack) node
rich            12.4.4   or later ※Required to run MOT(Norfair) node
```

*If the installation of cython_bbox fails on Windows, please try the installation from GitHub (as of 2022/06/05).<br>

```
pip install numpy
pip install Cython
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

# Installation
Please prepare the environment by one of the following methods.<br>
* Run the script directly
    1. Clone repository<br>`git clone https://github.com/Kazuhito00/Image-Processing-Node-Editor`
    1. Install package <br>`pip install -r requirements.txt`  
    1. Run "main.py" <br>`python main.py`
* Use Docker
    1. See [Image-Processing-Node-Editor/docker/nvidia-gpu](https://github.com/Kazuhito00/Image-Processing-Node-Editor/tree/main/docker/nvidia-gpu)
* Use executable file (Windows only)
    1. Download [ipn-editor_win_x86_64.zip](https://github.com/Kazuhito00/Image-Processing-Node-Editor/releases/download/v0.1.1/ipn-editor_win_x86_64.zip)
    1. Run "main.exe"
* Use pip installation<br><b>※The installed directory names will be "node" and "node_editor", so I plan to fix them in the future.<br>→When using pip, it is strongly recommended to install in a virtual environment such as venv.</b>
    1. Install build tools<br>Windows：https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/<br>Ubuntu：`sudo apt-get install build-essential libssl-dev libffi-dev python3-dev`
    1. Install Numpy, Cython, wheel<Br>`pip install Cython numpy wheel`
    1. Specify the GitHub repository and pip install<br>`pip install git+https://github.com/Kazuhito00/Image-Processing-Node-Editor`
    1. Start the application with the following command<br>`ipn-editor`  

# Usage
Here's how to run the app.
```bash
python main.py
```
* --setting<br>
Specifying the configuration file path that describes the node size and VideoWriter settings<br>
Default：node_editor/setting/setting.json
* --unuse_async_draw<br>
Do not use asynchronous drawing<Br>→Perform GUI event loop and node update process in series<br>*For investigating the cause of abnormal node termination, etc.<br>
Default：unspecified

### Create Node
Select the node you want to create from the menu and click<br>
<img src="https://user-images.githubusercontent.com/37477845/172030402-80d3d14e-d0c8-464f-bb0c-139bfe676845.gif" loading="lazy" width="50%">

### Connect Node
Drag the output terminal to connect to the input terminal<br>
Only the same type set for the terminal can be connected<br>
<img src="https://user-images.githubusercontent.com/37477845/172030403-ec4f0a89-22d5-4467-9b11-c8e595e65997.gif" loading="lazy" width="50%">

### Delete Node
With the node you want to delete selected, press the "Del" key<br>
<img src="https://user-images.githubusercontent.com/37477845/172030418-201d7df5-1984-4fa7-8e47-9264c5dcb6cf.gif" loading="lazy" width="50%">

### Export
Press "Export" from the menu and save the node settings(json file)<br>
<img src="https://user-images.githubusercontent.com/37477845/172030429-9c6c453c-b8b0-4ccf-b36e-eb666c2d919f.gif" loading="lazy" width="50%">

### Import
Read the node settings(json file) output by Export<br>
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
            Node that reads still images (bmp, jpg, png, gif) and outputs images<br>
            Open the file dialog with the "Select Image" button
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
            A node that reads a video (mp4, avi) and outputs an image for each frame<br>
            Open the file dialog with the "Select Movie" button<br>
            Check "Loop" to play the video in a loop<br>
            "Skip rate" sets the interval for skipping the output image.
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
            A node that reads a webcam and outputs an image for each frame<br>
            Specify the camera number in the Device No drop-down list<br>
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
            A node that reads the RTSP input of a network camera and outputs an image for each frame<br>
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
            Node that outputs an integer value<br>
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
            Node that outputs the float value<br>
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
            A node that applies pseudo color to the input image and outputs a pseudo color image
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
            A node that executes smoothing processing on the input image and outputs the smoothed image
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
            A node that executes brightness adjustment processing on the input image and outputs the brightness adjustment image<br>
            Brightness adjustment value can be changed with the "alpha" slide bar<br>
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
            A node that executes edge detection processing using the Canny method on the input image and outputs the edge detection image.<br>
           Specify the minimum and maximum thresholds with the slider
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
            A node that executes contrast adjustment processing on the input image and outputs the contrast adjustment image.<br>
           Contrast adjustment value can be changed with the "beta" slide bar<br>
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
            A node that performs cropping of the input image and outputs the cropped image<br>
            Upper left coordinates(x1, y1) and upper right coordinates(x2, y2) can be changed with the slider<br>
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
            Node that performs histogram flattening of the brightness part of the input image and outputs the image<br>
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
            A node that performs horizontal/vertical inversion to the input image and outputs the image<br>
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
            A node that performs gamma correction on the input image and outputs the image<br>
            Gamma value can be changed with the slider
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
            A node that grayscales the input image and outputs the image<br>
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
            A node that binarizes the input image and outputs the image<br>
            Specify the binarization algorithm with "type"<br>
            Change threshold with "threshold"<br><br>
            In "type", "Otsu binarization (THRESH_OTSU)" is an automatic threshold determination algorithm, so the "threshold" value is ignored.
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
            A node that performs 3x3 2D filtering processing on the input image and outputs the image
        </td>
    </tr>
</table>
</details>


<details>
<summary>Deep Learning Node</summary>

You can specify the model in the drop-down list and change the device at the time of inference with the CPU / GPU checkbox.<br>
* If the model does not support GPU inference, checking GPU will still result in CPU inference<br>
Refer to each directory of "node/deep_learning_node/XXXXXXXX" for the license of the model used by the node.
<table>
    <tr>
        <td width="200">
            Classification
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172043243-2c037f0b-e1ba-4e3b-96a8-b0e3358f6616.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            Node that performs classification on the input image<br>
            The output image is a raw image<br><br>
            Performs classification on the bounding box when an Object Detection node is connected
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
            Node that performs face detection on the input image<br>
            The output image is a raw image
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
            A node that performs Low-Light Image Enhancement on the input image<br>
            The output image is an image with Low-Light Image Enhancement applied.
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
            A node that performs monocular depth estimation on the input image<br>
            The output image is a grayscale image to which monocular depth estimation is applied.
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
            Node that performs object detection on the input image<br>
            The output image is a raw image
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
            Node that performs attitude estimation for the input image<br>
            The output image is a raw image
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
            Node that performs semantic segmentation on the input image<br>
            The output image is a raw image
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
            Node that executes QR code detection for the input image<br>
            The output image is a raw image
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
            A node that calculates FPS based on the processing time(ms) of the node<br>
           Processing time input terminal can be added with "Add Slot"
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
            Node that calculates the histogram of each RGB channel of the input image and displays it in the graph
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
            A node that evaluates image quality using BRISQUE<br>
            * The higher the number, the worse
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
            Draw the analysis result for the image of the node that outputs the raw image such as Classification node and Object Detection node.
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
            Node that displays multiple input images side by side<br>
           Image input terminal can be added with "Add Slot"
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
            A node that draws text in the upper left of the input image<br>
            Drawing color can be selected in the color map<br>
            By connecting the processing time input terminal, the processing time is also drawn.
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
            Node to display the image<br>
            Display larger than the processing node<br>
            Also, if you connect a node that outputs raw images such as a Classification node or Object Detection node, the analysis result will be added and drawn.
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
            Larger than the Result Image node
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
            Node to switch whether to output the input image or not
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
            Node to export the input image as a video<br>
            Output destination, output size, FPS are specified in "setting.json"
        </td>
    </tr>
</table>
</details>

<details>
<summary>Preview Release Node</summary>

Nodes whose specifications may change significantly in the future
<table>
    <tr>
        <td width="200">
            MOT
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172049681-67df2cc3-3db3-4766-a96e-f7c557e4a5b9.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that inputs an Object Detection node and executes MOT(Multi Object Tracking)
        </td>
    </tr>
</table>
</details>

# ToDo
- [ ] Investigating the problem that the graph part of the RGB Histgram node is always in the foreground
- [ ] Investigating the problem that the connection line remains when deleting a node that connects multiple nodes
- [ ] Improved behavior that the import feature can only be used before adding a node
  
# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
Image-Processing-Node-Editor is under [Apache-2.0 license](LICENSE).<br><br>
IThe source code of mage-Processing-Node-Editor itself is [Apache-2.0 license](LICENSE), but <br>
The source code for each algorithm is subject to its own license. <br>
For details, please check the LICENSE file included in each directory.

# License(Image)
The images displayed in the sample are borrowed from the [Free Material Pakutaso](https://www.pakutaso.com/),and  [NHK Creative Library](https://www.nhk.or.jp/archives).
