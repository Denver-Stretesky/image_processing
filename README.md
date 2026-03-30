# Image Processing

## Models That can be Used
Here are the download links to the models used:

U2net: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

ISnet: https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx

Birefnet: model is too large to practically use
## Notes
Used through the command line at the moment. There still needs to be cleanup of the code before it is ready to be integrated

## Steps
### File Validation
Determine if the input file is a valid file type. Then determine how many channels/what the channels are. This information will be used now to determine if background removal is necessary or if the image is cmyk. (nothing is done with cmyk yet but it needs a special conversion so the colours look better)

### Background Removal
First check if the image has no background. If it does then skip background removal. Then if the object is on a white background skip background removal since the the model struggles with it and simple contours works better.

### Contour extraction
Finds the object in the image. This will then be used to genrate the dimensions of the object for use in sizing

### Compositing
Places the object in the center of an appropriatly sized white background

### Resizing
First resizes up if lower than desired size ising integer scaling. This is done so that a more advanced neural network based upscaler can be used at a later date. Then it gets scaled down to the desired dimensions.

### Saving
Saved as a png. Instead, MatToBytes can be used to format in the way image api requires
