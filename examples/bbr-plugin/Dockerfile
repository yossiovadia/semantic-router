FROM golang:1.25-bookworm AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /bbr-server ./cmd/server/

FROM gcr.io/distroless/static-debian12
COPY --from=builder /bbr-server /bbr-server
EXPOSE 9002
ENTRYPOINT ["/bbr-server"]
CMD ["-port", "9002"]
