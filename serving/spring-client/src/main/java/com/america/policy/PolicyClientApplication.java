package com.america.policy;

import com.america.ai.v1.PolicyAssistantGrpc;
import com.america.ai.v1.PolicyQuestion;
import com.america.ai.v1.StoreContext;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.MetadataUtils;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

@SpringBootApplication
public class PolicyClientApplication {
    public static void main(String[] args) { SpringApplication.run(PolicyClientApplication.class, args); }

    @Bean(destroyMethod = "shutdown")
    ManagedChannel policyChannel(@Value("${ai.grpc-target:127.0.0.1:50051}") String target) {
        // Plaintext is for localhost/SSH-tunnel/private-network deployment only.
        return ManagedChannelBuilder.forTarget(target).usePlaintext().build();
    }

    public record Store(String storeId, String name, String category, String region, String address) {}
    public record Question(String question, Store store) {}
    public record Source(String title, String url, String snippet) {}
    public record Answer(String answer, List<Source> sources) {}

    @RestController
    public static class PolicyController {
        private final PolicyAssistantGrpc.PolicyAssistantBlockingStub stub;

        public PolicyController(ManagedChannel channel, @Value("${AI_INTERNAL_KEY}") String key) {
            if (key.isBlank()) throw new IllegalArgumentException("AI_INTERNAL_KEY is required");
            Metadata metadata = new Metadata();
            metadata.put(Metadata.Key.of("x-internal-key", Metadata.ASCII_STRING_MARSHALLER), key);
            stub = PolicyAssistantGrpc.newBlockingStub(channel)
                    .withInterceptors(MetadataUtils.newAttachHeadersInterceptor(metadata));
        }

        @PostMapping("/api/policy/questions")
        public Answer ask(@RequestBody Question body) {
            if (body.question() == null || body.question().isBlank() || body.question().length() > 20000)
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "question is required (max 20000 characters)");
            Store s = body.store() == null ? new Store("", "", "", "", "") : body.store();
            StoreContext store = StoreContext.newBuilder().setStoreId(value(s.storeId())).setName(value(s.name()))
                    .setCategory(value(s.category())).setRegion(value(s.region())).setAddress(value(s.address())).build();
            try {
                var result = stub.withDeadlineAfter(90, TimeUnit.SECONDS).ask(
                        PolicyQuestion.newBuilder().setStore(store).setQuestion(body.question()).setTopK(3).build());
                return new Answer(result.getAnswer(), result.getSourcesList().stream()
                        .map(x -> new Source(x.getTitle(), x.getUrl(), x.getSnippet())).toList());
            } catch (StatusRuntimeException e) {
                HttpStatus status = switch (e.getStatus().getCode()) {
                    case INVALID_ARGUMENT -> HttpStatus.BAD_REQUEST;
                    case RESOURCE_EXHAUSTED -> HttpStatus.TOO_MANY_REQUESTS;
                    case DEADLINE_EXCEEDED -> HttpStatus.GATEWAY_TIMEOUT;
                    default -> HttpStatus.SERVICE_UNAVAILABLE;
                };
                throw new ResponseStatusException(status, "Policy AI unavailable: " + e.getStatus().getCode());
            }
        }
        private static String value(String text) { return text == null ? "" : text; }
    }
}
