kernel_code = """
__global__ void resize(float *output, const float *input,
                        const int channels, const int width, const int height,
                        const int new_width, const int new_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height) {
        float x_ratio = (float)(width - 1) * x / (new_width - 1);
        float y_ratio = (float)(height - 1) * y / (new_height - 1);

        int x0 = (int)x_ratio;
        int y0 = (int)y_ratio;
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);

        float x_diff = x_ratio - x0;
        float y_diff = y_ratio - y0;

        for (int c = 0; c < channels; ++c) {
            float top_left = input[(y0 * width + x0) * channels + c];
            float top_right = input[(y0 * width + x1) * channels + c];
            float bottom_left = input[(y1 * width + x0) * channels + c];
            float bottom_right = input[(y1 * width + x1) * channels + c];

            float top = top_left * (1 - x_diff) + top_right * x_diff;
            float bottom = bottom_left * (1 - x_diff) + bottom_right * x_diff;

            output[(y * new_width + x) * channels + c] = top * (1 - y_diff) + bottom * y_diff;
        }
    }
}

__global__ void center_crop(float *output, const float *input, 
                            const int channels, const int width, const int height,
                            const int crop_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < crop_size && y < crop_size) {
        int start_x = (width - crop_size) / 2;
        int start_y = (height - crop_size) / 2;

        for (int c = 0; c < channels; ++c) {
            output[(y * crop_size + x) * channels + c] =
                input[((start_y + y) * width + start_x + x) * channels + c];
        }
    }
}

__global__ void transpose(float *output, const float *input, const int channels, const int width, const int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && c < channels) {
        output[c * width * height + y * width + x] = input[y * width * channels + x * channels + c];
    }
}

__global__ void normalize(float* output, const float* input, const int size) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size){

      size_t i_idx0 = tid;
      size_t i_idx1 = tid + size;
      size_t i_idx2 = tid + 2 * size;


      output[i_idx0] = ( input[i_idx0] - 0.485 ) / 0.229;
      output[i_idx1] = ( input[i_idx1] - 0.456 ) / 0.224;
      output[i_idx2] = ( input[i_idx2] - 0.406 ) / 0.225;
    }
}

"""