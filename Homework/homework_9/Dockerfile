FROM agrigorev/model-2025-hairstyle:v1

# Install needed Python libraries
# (Some may already be there, but this is safe)
RUN pip install --no-cache-dir numpy onnxruntime pillow requests

# Copy our lambda handler code
COPY lambda_function.py .

# Tell AWS Lambda which function to call
CMD ["lambda_function.lambda_handler"]
