#pragma once

#ifdef __cplusplus
extern "C" {
#endif
    struct CNN;
    struct CNN* torch_initialize_model(char const* model, char const* options);
    void   torch_delete_model(struct CNN* model);
    void   torch_training(struct CNN* model, float* data, int* target, size_t data_size, int num_epochs);
    size_t torch_inference(struct CNN* model, float* data, size_t data_size, float* result_buffer, size_t buffer_size);
    size_t torch_model_output_size(struct CNN const* model);
    size_t torch_model_input_size(struct CNN const* model);
#ifdef __cplusplus
}
#endif
