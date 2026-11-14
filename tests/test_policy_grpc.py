import threading

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from america.ai.v1 import common_pb2, policy_pb2, policy_pb2_grpc
from policy_r1.grpc_service import build_server
from policy_r1.web_search import SearchUnavailable


class Engine:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False
        self.search_error = None

    def answer(self, question, **kwargs):
        if self.search_error is not None:
            raise self.search_error
        self.calls.append((question, kwargs))
        self.entered.set()
        if self.block:
            self.release.wait(5)
        if self.fail:
            raise SearchUnavailable("provider detail")
        return {"answer": "공식 공고를 확인해 주세요.", "searches": 1,
                "sources": [{"title": "공고", "url": "https://example.go.kr/policy", "snippet": "안내", "score": 1}]}


@pytest.fixture
def rpc():
    engine = Engine()
    server, port = build_server(engine, "dummy-internal-key", "127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=3)
    try:
        yield engine, policy_pb2_grpc.PolicyAssistantStub(channel), channel
    finally:
        engine.release.set()
        channel.close()
        server.stop(0).wait()


AUTH = (("x-internal-key", "dummy-internal-key"),)


def request():
    return policy_pb2.PolicyQuestion(question="지원 정책", top_k=3,
        store=common_pb2.StoreContext(name="private store", region="서울", category="음식점"))


def test_real_grpc_contract_auth_health_and_stream(rpc):
    engine, stub, channel = rpc
    with pytest.raises(grpc.RpcError) as error:
        stub.Ask(request(), timeout=3)
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED and engine.calls == []
    answer = stub.Ask(request(), metadata=AUTH, timeout=3)
    assert answer.sources[0].url == "https://example.go.kr/policy"
    assert engine.calls[0][1]["region"] == "서울"
    assert "name" not in engine.calls[0][1]
    chunks = list(stub.AskStream(request(), metadata=AUTH, timeout=3))
    assert "".join(chunk.delta for chunk in chunks) == answer.answer
    assert chunks[-1].done and chunks[-1].sources[0] == answer.sources[0]
    health = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=3)
    assert health.status == health_pb2.HealthCheckResponse.SERVING


def test_gpu_busy_and_search_failure(rpc):
    engine, stub, _ = rpc
    engine.block = True
    active = stub.Ask.future(request(), metadata=AUTH, timeout=5)
    assert engine.entered.wait(2)
    with pytest.raises(grpc.RpcError) as busy:
        stub.Ask(request(), metadata=AUTH, timeout=3)
    assert busy.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    engine.release.set()
    active.result(timeout=3)
    engine.block, engine.fail = False, True
    with pytest.raises(grpc.RpcError) as failed:
        stub.Ask(request(), metadata=AUTH, timeout=3)
    assert failed.value.code() == grpc.StatusCode.UNAVAILABLE
    assert "provider detail" not in failed.value.details()


@pytest.mark.parametrize("code,expected", [("TIMEOUT", "TIMEOUT"), ("HTTP_429", "HTTP_429"),
                                         ("secret-key-value", "SEARCH_ERROR")])
def test_safe_search_error_code_reaches_client_without_credentials(rpc, caplog, code, expected):
    engine, stub, _ = rpc
    engine.search_error = SearchUnavailable("secret-key-value", code=code, search_id="a" * 24)
    with pytest.raises(grpc.RpcError) as failure:
        stub.Ask(request(), metadata=AUTH, timeout=3)
    assert failure.value.code() == grpc.StatusCode.UNAVAILABLE
    assert f"[{expected}]" in failure.value.details() and "search_id=" in failure.value.details()
    assert "secret-key-value" not in failure.value.details() and "secret-key-value" not in caplog.text
