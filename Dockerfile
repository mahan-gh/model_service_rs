FROM rust:1.72 AS builder

RUN apt-get update && apt-get install -y \
  pkg-config \
  libssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Build the application
RUN cargo build --release

# Find the libtensorflow libraries in the build output
RUN find target/release/build -name "libtensorflow*.so*" -exec cp {} target/release/ \;

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
  libssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the release binary and TensorFlow libraries
COPY --from=builder /app/target/release/model_service_rs /app/
COPY --from=builder /app/target/release/libtensorflow*.so* /app/

# Copy model files from local directory
COPY ./model/class_list.txt /app/model/
COPY ./model/frozen_graph.pb /app/model/

# Add the current directory to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

EXPOSE 8080

# Set the entry point for the application
CMD ["./model_service_rs"]