"""Publish notification batches for manual end-to-end throughput testing."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    from app.services.messaging.interface.producer import IMessagingProducer

from app.services.messaging.config import (
    MessageBrokerType,
    RedisStreamsConfig,
    Topic,
    get_message_broker_type,
)
from app.services.messaging.kafka.config.kafka_config import KafkaProducerConfig
from app.services.messaging.messaging_factory import MessagingFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a batch and report broker-publish throughput.",
    )
    parser.add_argument("--org-id", required=True, help="Recipient organization ObjectId")
    parser.add_argument("--user-id", required=True, help="Recipient user ObjectId")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def env_bool(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.lower() in {"1", "true", "yes"}


def create_producer(logger: logging.Logger) -> IMessagingProducer:
    broker_type = get_message_broker_type()
    if broker_type == MessageBrokerType.KAFKA:
        brokers = [
            broker.strip()
            for broker in os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
            if broker.strip()
        ]
        username = os.getenv("KAFKA_USERNAME")
        password = os.getenv("KAFKA_PASSWORD")
        sasl = None
        if username and password:
            sasl = {
                "mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
                "username": username,
                "password": password,
            }
        config = KafkaProducerConfig(
            bootstrap_servers=brokers,
            client_id="notification-throughput-benchmark",
            ssl=env_bool("KAFKA_SSL"),
            sasl=sasl,
        )
    else:
        config = RedisStreamsConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=int(os.getenv("REDIS_DB", "0")),
            max_len=int(os.getenv("REDIS_STREAMS_MAXLEN", "500000")),
            client_id="notification-throughput-benchmark",
        )

    return MessagingFactory.create_producer(
        logger=logger,
        config=config,
        broker_type=broker_type,
    )


def build_notification(
    *,
    org_id: str,
    user_id: str,
    benchmark_id: str,
    benchmark_started_at_ms: int,
    sequence: int,
    total: int,
) -> dict[str, Any]:
    return {
        "orgId": org_id,
        "type": "CONNECTOR_INFO",
        "severity": "info",
        "status": "unread",
        "originService": "Connector Service",
        "title": f"Notification throughput test {sequence}/{total}",
        "message": f"Benchmark {benchmark_id}",
        "payload": {
            "send_time_ms": int(time.time() * 1000),
            "benchmark_id": benchmark_id,
            "benchmark_sequence": sequence,
            "benchmark_total": total,
            "benchmark_started_at_ms": benchmark_started_at_ms,
        },
        "recipientUserIds": [user_id],
        "recipientRoles": [],
        "isDeleted": False,
    }


async def run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("notification-throughput-benchmark")
    producer = create_producer(logger)
    await producer.initialize()

    benchmark_id = str(uuid.uuid4())
    started_at_ms = int(time.time() * 1000)
    semaphore = asyncio.Semaphore(min(args.concurrency, args.count))

    async def publish(sequence: int) -> None:
        async with semaphore:
            notification = build_notification(
                org_id=args.org_id,
                user_id=args.user_id,
                benchmark_id=benchmark_id,
                benchmark_started_at_ms=started_at_ms,
                sequence=sequence,
                total=args.count,
            )
            await producer.send_message(
                topic=Topic.NOTIFICATION.value,
                message=notification,
                key=f"{benchmark_id}-{sequence}",
            )

    started = time.perf_counter()
    try:
        await asyncio.gather(*(publish(sequence) for sequence in range(1, args.count + 1)))
    finally:
        await producer.cleanup()

    elapsed_seconds = time.perf_counter() - started
    per_second = args.count / elapsed_seconds

    logger.warning("Benchmark ID:       %s", benchmark_id)
    logger.warning("Published:          %d", args.count)
    logger.warning("Publish duration:   %.3fs", elapsed_seconds)
    logger.warning("Publish throughput: %.2f/second", per_second)
    logger.warning("Publish throughput: %.2f/minute", per_second * 60)
    logger.warning("Browser delivery results will appear in the frontend console.")


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
