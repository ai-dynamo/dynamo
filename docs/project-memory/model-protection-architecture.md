---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Secure model protection architecture
---

## Kiến trúc bảo vệ model cho Dynamo

Last updated: 2026-10-07 (Asia/Ho_Chi_Minh).

### Runtime vLLM 0.30.0

Protected runtime hiện khóa chính xác vLLM `0.30.0` và base Omni `0.30.0rc1`.
API `AsyncEngineArgs`, normalized `VllmConfig` và entrypoint `EngineCoreProc`
được kiểm tra trên base image vLLM `0.30.0`. Chỉ nhận `auto`/`safetensors`;
không bật loader `ipc_cache` hoặc daemon lưu weights khi nâng phiên bản.
Nghiệm thu GPU/model/TPM trên image cuối vẫn là release gate. Những kết quả
vLLM 0.28/0.29 bên dưới là lịch sử, không phải bằng chứng cho image mới.

### Cập nhật profile có layer

Runtime hiện hỗ trợ JSON V2 với bốn cờ boolean mặc định false:
`package_verification`, `license_verification`, `tpm_binding`,
`secure_materialization`. Không có ba cờ mới cho hardening/engine/audit;
các giới hạn an toàn hiện có vẫn áp dụng cho mọi session mã hóa.

Manifest V2 ký profile `encrypted-file`, `encrypted-file-license` hoặc
`encrypted-tpm`; các đoạn baseline TPM bên dưới chỉ áp dụng cho profile TPM.
Manifest V1 không khai báo profile được coi là TPM. Không đổi profile bằng
env cho package đã ký. Chế độ phần mềm dùng một DEK riêng của package trong
file `0600`, không đưa issuer KEK/private signer lên máy khách. License phần
mềm có signature domain riêng, không dùng recipient TPM giả. Model plain
được phát hiện trước khi đọc JSON và đi theo luồng cũ.

Xem [hướng dẫn profile và bàn giao](Protected-Model-Usage.md) mục 0 để chạy
chế độ không TPM. Hai profile phần mềm không có device binding; đây là
giới hạn bảo mật chủ động, không phải lỗi TPM được bỏ qua.

Hướng dẫn thao tác riêng cho profile chỉ mã hóa là
[model-encryption-only.md](model-encryption-only.md); bản tiếng Anh là
[model-encryption-only.en.md](model-encryption-only.en.md). Hướng dẫn này mô tả
đóng gói trên máy phát hành, tạo `runtime.json`, bàn giao `package/` và
`runtime/`, rồi cấu hình Compose trên server khách.

#### Phân phối metadata giữa worker và frontend

Worker protected giải mã package vào session riêng dưới tmpfs. Đường dẫn này
chỉ tồn tại trên worker; frontend không thể mở nó trực tiếp. Với triển khai
Compose có frontend và worker tách container, worker cần bật system status
server bằng `DYN_SYSTEM_PORT` và self-host metadata bằng
`DYN_SELF_HOST_METADATA=true`. Frontend và worker phải cùng Docker network để
frontend truy cập endpoint nội bộ của worker. Không cần publish system port ra
host.

```mermaid
sequenceDiagram
    participant C as Compose worker
    participant W as Protected worker
    participant F as Frontend
    participant HF as Hugging Face
    C->>W: DYN_SYSTEM_PORT=9090; self-host metadata=true
    W->>W: Verify package; decrypt to private tmpfs session
    F->>W: Fetch declared config/tokenizer metadata over internal HTTP
    W-->>F: Return metadata from the active session
    Note over F,HF: Successful self-hosting avoids remote fallback
```

Nếu system endpoint thiếu hoặc không thể truy cập, frontend có thể nhận đường
dẫn session như `/run/protected-ocr-models/<id>` rồi dùng nó như Hugging Face
repository ID. URL dạng `/api/models//run/protected-ocr-models/...` là dấu hiệu
metadata resolution đã fallback, không phải bằng chứng package mã hóa bị tải
lên Hugging Face. Dòng `ignore_weights=true` cũng không ngăn tải metadata.
Khắc phục ở cấu hình worker/network; không thay đường dẫn model thành repo ID
giả và không tắt kiểm tra package.

Với Compose protected OCR/LLM hiện tại, cấu hình worker đặt
`DYN_SYSTEM_PORT: "9090"`, `DYN_SELF_HOST_METADATA: "true"`, `HOME: /tmp` và
`CUPY_CACHE_DIR: /tmp/cache/cupy`. Cache CuPy là vấn đề ghi filesystem riêng,
không thuộc mã hóa model. Thay đổi Compose chỉ cần recreate worker; chỉ cần
build lại image nếu image không có loader hoặc bản sửa code cần thiết. Cấu hình
đã qua kiểm tra cú pháp Compose tại máy phát hành; xác minh endpoint và inference
trên server khách vẫn đang mở.

Tài liệu này là kiến trúc V2 sau
[security review](model-protection-security-review.md). Thiết kế giữ nguyên
luồng model bình thường, bảo vệ model được phân phối dưới dạng mã hóa và dùng
TPM để ràng buộc khả năng giải mã với máy đã đăng ký.

Baseline V1 giả định hệ thống on-prem có thể air-gapped:

- license offline perpetual, hỗ trợ reactivation khi thay máy;
- TPM 2.0 giữ private unwrap key không export được;
- mỗi customer/model version có một immutable encrypted artifact và DEK mới;
- plaintext model chỉ xuất hiện trong dedicated tmpfs;
- một engine aggregated nằm trong một Pod/node; V1 product slice hiện tại chỉ
  cho `TP=PP=DP=1`;
- chỉ hỗ trợ safetensors và các đường load đã nằm trong allowlist; multimodal
  chỉ được bật khi processor/encoder metadata nằm trong manifest;
- USB dongle và HSM/PKCS#11 không thuộc baseline V1;
- EK certificate/AK vendor attestation được defer khỏi V1 hiện tại; enrollment
  dùng development signer và chỉ phù hợp pilot có kiểm soát;
- private issuer keys nằm ngoài source/image/config, trong encrypted offline
  key directory chỉ mount khi chạy packager/issuer.

Calendar expiry, renewal và revocation nhanh cần online key-release profile.
Online profile được thiết kế ở phase sau, không tạo provider abstraction trước
khi có yêu cầu kết nối cụ thể.

> [!IMPORTANT]
> Yêu cầu sản phẩm: protected loading phải dùng đúng topology TP/PP/DP và số
> worker do deployment cấu hình. Không giữ giới hạn cố định
> `TP=PP=DP=1` và không ràng buộc license với số GPU. Bootstrap hiện vẫn từ
> chối giá trị khác 1. Nghiệm thu multi-GPU còn mở; không công bố đã hỗ trợ cho
> tới khi worker entry, quyền sở hữu session dùng chung, lỗi một phần rank,
> restart và cleanup được kiểm tra với topology cấu hình. Profile một GPU bên
> dưới là baseline triển khai hiện tại, không phải mục tiêu scale của sản phẩm.

### Trạng thái triển khai hiện tại

> [!WARNING]
> Hardening R-01–R-18 đã được triển khai trong working tree ngày 2026-09-13.
> Rust/CLI tests, Clippy, PyO3 TPM build, focused Python tests và local physical
> TPM → tmpfs → vLLM 0.28 → RTX 3060 inference đã pass; software issuer CLI đã thay PKCS#11,
> nhưng key-directory operations, cross-server và full CI/DEP/security approval
> vẫn là release gate. EK/AK vendor attestation được defer khỏi V1 hiện tại;
> Kubernetes chỉ là gate riêng
> nếu release tuyên bố hỗ trợ Kubernetes.
> Xem [task ledger](TASKS.md) và mục 6–7 của
> [security review](model-protection-security-review.md).

> [!NOTE]
> Ngày 2026-10-07, tài liệu profile `encrypted-file` đã được bổ sung song ngữ.
> Cấu hình Compose OCR/LLM hiện nêu system metadata endpoint và cache CuPy có
> thể ghi. Đây mới là sửa cấu hình được lint/kiểm tra cú pháp; log khởi động và
> OCR/LLM inference trên server khách chưa được xác nhận.

Crate `lib/model-protection` đã có strict manifest parser, domain-separated
exact-byte Ed25519 verification cho package và license bằng hai trust root khác nhau,
AES-256-GCM record codec, runtime-version floor, opaque authorization/key
handles, namespace validation và Linux `openat2` materialization vào tmpfs.
Session owner đã có exclusive lock, UUID session, stale-session cleanup,
tmpfs/cgroup preflight, pinned session-directory FD và RAII cleanup. Raw decrypt API không được export khỏi
crate. Error display/debug/source chỉ trả stable sanitized code; reason nội bộ
là chuỗi tĩnh allowlisted.

Rust core hiện đã nối vào PyO3 và bootstrap vLLM hai phase. Protected mode
stage metadata trước, kiểm tra raw/effective vLLM configuration, chỉ sau đó mới
gọi TPM và materialize weights; plain mode trả nguyên argv/path. Mutable engine
routes bị tắt trong protected mode và session được giữ đến worker shutdown.
Deployment profile riêng dưới `deploy/model-protection/` mount memory-backed
`emptyDir`; runtime path luôn là `/run/<validated-DYN_NAMESPACE>-models` (ví dụ
`DYN_NAMESPACE=protected` tạo `/run/protected-models`) và schedule lên node TPM.

Module ESAPI TPM unwrap có dưới feature opt-in `tpm2`. `PolicyAuthorize`
authPolicy hiện dùng đúng two-hash construction và có frozen expected vector
độc lập; PyO3 với TPM feature đã compile/link trong container có `tpm2-tss`.
Actual issuer → physical TPM ESAPI unwrap đã được chạy trên host này; swtpm
vector là historical regression evidence. V1 hiện tại cố ý dùng development
enrollment; EK/AK-certified production enrollment là phase nâng cấp sau.
Software packager/issuer dùng encrypted PKCS#8/AES-KWP/RSA-OAEP software keys.
Format package, signed issuer-record V2, license và
TPM recipient không đổi. Packager dùng fresh DEK, stream-encrypt, ký manifest,
AES-KWP-wrap DEK bằng issuer KEK; issuer verify exact association, unwrap DEK
trong memory khóa, RSA-OAEP-wrap tới public DUK, ký TPM policy và license rồi
zeroize. Key material chỉ đọc từ absolute regular files trong một encrypted
offline key directory, không từ CLI value, environment, image hoặc repository.
Chưa có fresh EK/AK/DUK enrollment service, target-cluster admission,
crash/Pod-restart hoặc cross-server evidence. [DEP #14764](https://github.com/ai-dynamo/dynamo/issues/14764)
và security approval vẫn là release gate; vì vậy chưa được gọi là product-ready.

## 1. Mục tiêu và threat model

### 1.1 Bảo đảm của V1

- Model bình thường tiếp tục chạy theo Dynamo/vLLM hiện tại.
- Copy encrypted package và license sang máy khác không đủ để lấy DEK.
- Thay package, manifest, license hoặc ciphertext bị phát hiện trước khi vLLM
  sử dụng model.
- Plaintext weights không được ghi vào filesystem thường, image layer, cache,
  log, crash dump, swap hoặc hibernation image trong secure deployment profile.
- Key, cipher state và plaintext staging có owner và cleanup path xác định.
- Secure failure luôn dừng startup; không fallback sang plain loader.

### 1.2 Giới hạn bảo vệ

| Attacker/tình huống | Bảo đảm |
|---|---|
| Chỉ có package/license (TPM profile) | Không decrypt được nếu không có TPM đã đăng ký |
| Sửa source package | Signature, strict schema, hash hoặc AEAD phát hiện |
| Process khác UID, không có quyền quản trị | Filesystem permissions và Pod isolation chặn đọc session |
| Process cùng UID hoặc có quyền `pods/exec`/debug | Có thể đọc plaintext tmpfs; thuộc trusted operator boundary |
| Host root, privileged container, kernel, `ptrace`, GPU dump | Ngoài guarantee chống extract của V1 |
| License bị thu hồi sau khi weights đã bị copy | Không thể thu hồi bản copy đã bị extract |

TPM V1 cung cấp machine-bound key custody. Nó không chứng minh worker process
không bị patch và không tự hiểu claim trong `license.json`. Vì root trên máy đã
được cấp phép nằm ngoài threat model, offline V1 không được quảng bá là DRM
chống administrator của chính máy đó.

### 1.3 Trusted computing base

Trusted boundary gồm:

- build/signing environment;
- offline issuer host và encrypted key directory giữ issuer-side DEK record;
- TPM, firmware và boot chain theo deployment policy;
- signed runtime image và model-protection Rust core;
- Kubernetes/node administrator có quyền đổi Pod, mount, UID hoặc debug;
- kernel, container runtime, GPU driver và vLLM version đã duyệt.

Không đặt sidecar hoặc ephemeral container không tin cậy trong cùng security
domain với secure worker. Mỗi secure worker dùng dedicated UID và volume.

## 2. Hai luồng model

### 2.1 Plain model

```text
model source
    ↓
bounded secure-marker detection
    ↓
PLAIN
    ↓
existing Dynamo fetch/config/vLLM path
    ↓
GPU
```

Plain path không verify license, không lấy key, không tạo secure session và
không đổi model source. Nếu secure volume từ một run cũ hiện diện, stale-volume
recovery là hoạt động của workload lifecycle; nó không biến plain model thành
secure model và không được làm plain startup phụ thuộc TPM.

### 2.2 TPM-bound protected model

```text
local encrypted package + separate signed license
    ↓
secure bootstrap
    ↓
manifest/license/package verification
    ↓
verified public metadata staged trong tmpfs
    ↓
effective vLLM config validation
    ↓
TPM-authorized DEK unwrap
    ↓
authenticated record decrypt vào private tmpfs model view
    ↓
vLLM load + Dynamo registration
    ↓
serving
    ↓
stop/reap engine children + cleanup session
```

#### Vai trò của từng layer

| Layer | Dữ liệu vào | Layer làm gì | Kết quả hoặc failure behavior |
|---|---|---|---|
| **Local encrypted package + separate signed license** | Package chứa ciphertext, signed manifest và public metadata; license được phát hành riêng | Package giữ artifact model bất biến. License giữ entitlement, package digest, device binding và wrapped DEK để có thể reissue khi thay máy mà không đóng gói lại model. | Cung cấp hai artifact có lifecycle độc lập. Package hoặc license thiếu marker bắt buộc thì input là secure-invalid. |
| **Secure bootstrap** | Model argument, license path, runtime options và effective namespace inputs | Chạy trước khi import vLLM/tạo `EngineArgs`; phân loại plain/protected, khóa plugin allowlist, đặt core-dump/dumpable/no-swap process policy, chặn sớm mode không hỗ trợ, xác định namespace, giành quyền sở hữu volume và tạo secure session. Trước key release, raw config chặn EC connector ngoài; runtime tạo connector multimodal được allowlist rồi mới tạo/kiểm tra `VllmConfig`. Layer này không decrypt weights. | Plain model quay về luồng hiện có mà không chạy host/key/license checks. Protected model chỉ đi tiếp khi source là local, persistence policy đạt, session thuộc đúng owner và dedicated tmpfs hợp lệ. |
| **Manifest/license/package verification** | Exact manifest bytes, signature envelope, license, trust roots và package entries | Verify chữ ký package/license, strict schema, package digest, model/customer/product identity, device DUK binding, file inventory, size/count limits và anti-downgrade fields. | Tạo verified package descriptor. Bất kỳ signature, schema, binding hoặc bounds check nào sai đều dừng trước khi lấy DEK và trước khi tạo plaintext. |
| **Verified public metadata staged trong tmpfs** | Các file public được manifest allowlist như `config.json`, tokenizer và safetensors index | Đọc bằng component-safe path, bounded-copy vào private staging, hash chính bytes đã đọc và chỉ atomic-publish khi khớp manifest. License, wrapped key và session state không nằm trong model view. | Tạo metadata đã xác thực trực tiếp trong UUID session directory được truyền cho vLLM. Symlink, source mutation, undeclared file, hash/size mismatch hoặc path traversal làm cleanup session và fail closed. |
| **Effective vLLM config validation** | vLLM config đã parse/normalize từ internal rewritten arguments | Kiểm tra lại model, tokenizer, config, load format, plugin và worker topology sau khi vLLM áp dụng default hoặc rewrite. Mọi path phải trỏ vào verified model view hoặc signed allowlist; runtime reload surface phải bị gate. | Sinh effective config được phép chạy. Mode như remote loader, Ray/multi-node, snapshot, dynamic LoRA, runtime weight update hoặc `trust_remote_code` bị từ chối trước key release. |
| **TPM-authorized DEK unwrap** | Signed license, wrapped DEK, certified Device Unwrap Key (DUK) và TPM authorization policy | Xác nhận license/package/device binding lần cuối, yêu cầu TPM dùng private DUK không export được để unwrap DEK theo policy đã duyệt, rồi đưa DEK vào owned Rust secret buffer. | Cung cấp DEK tạm thời cho decryptor. Sai device, sai policy, TPM reset, wrapped key bị sửa hoặc unwrap lỗi đều dừng mà không có file weight plaintext. |
| **Authenticated record decrypt vào private tmpfs model view** | Verified ciphertext records, DEK, nonce/counter, Additional Authenticated Data (AAD) và signed output metadata | Chạy ngoài asyncio loop và release Python GIL. Decrypt từng AES-256-GCM record; chỉ ghi plaintext của record sau khi tag hợp lệ, kiểm cancellation giữa record và trước write. Ghi vào file `.partial`, kiểm tra thứ tự, exact EOF, full-file size/hash, rồi atomic rename sang tên cuối. | Tạo Hugging Face-compatible weights trong UUID session directory. Sai tag, cancellation, record reorder/replay, truncation, append, hash/size mismatch, ENOSPC hoặc ENOMEM xóa toàn bộ session và zeroize owned DEK buffers; cleanup chỉ chạy sau khi writer đã join. |
| **vLLM load + Dynamo registration** | Verified effective config và private tmpfs model path | vLLM load weights lên CPU/GPU theo allowlist; Dynamo đăng ký đúng served identity và chỉ publish declared public metadata. Session owner theo dõi engine children và registration state. | Chỉ chuyển sang ready khi cả engine initialization và Dynamo registration thành công. Lỗi ở một trong hai bước sẽ stop/reap child processes trước khi cleanup. |
| **Serving** | Engine đã ready và model đã đăng ký | Phục vụ inference trong topology V1 đã duyệt. tmpfs model view được giữ suốt worker lifetime vì chưa chứng minh mọi đường vLLM/Dynamo ngừng đọc lại file sau startup. | Request mới chỉ được nhận sau readiness. Secure mode tiếp tục chặn các API reload/LoRA/weight-update ngoài allowlist. |
| **Stop/reap engine children + cleanup session** | Shutdown signal, startup failure, registration failure hoặc worker termination | Ngừng admission, dừng và reap toàn bộ process con do session owner quản lý, đóng handles, xóa model/state/session, giải phóng lock và zeroize các secret buffer còn sở hữu. Startup kế tiếp dọn stale session của owner cũ. | Không để plaintext session còn có thể truy cập sau graceful shutdown. Với `SIGKILL`, cleanup được thực hiện ở lần startup kế tiếp hoặc khi Pod bị xóa; không claim physical RAM zeroization. |

Thứ tự các layer là một security invariant:

1. Verify package, license, paths, resource bounds và effective vLLM config
   trước khi yêu cầu TPM unwrap DEK.
2. Authenticate từng ciphertext record trước khi publish plaintext tương ứng.
3. Chỉ công bố readiness sau khi vLLM load và Dynamo registration cùng thành
   công.
4. Giữ một session owner chịu trách nhiệm từ lúc tạo tmpfs session đến khi
   engine children đã dừng và session được xóa.

Một reserved marker xuất hiện thì input là `SECURE_CANDIDATE` hoặc
`SECURE_INVALID`; không được downgrade sang plain.

### 2.3 Software file-key profiles

`encrypted-file` và `encrypted-file-license` dùng chung bước nhận diện package,
xác thực chữ ký, kiểm tra metadata và cấu hình vLLM, materialize vào tmpfs,
đăng ký worker và cleanup session với profile TPM. Hai profile này chỉ thay đổi
bước cấp quyền và lấy khóa:

```text
signed encrypted package
    ↓
secure bootstrap + profile/layer validation
    ↓
package signature and metadata verification
    ↓
effective vLLM config validation
    ↓
[optional software-license verification]
    ↓
read package-specific 32-byte DEK from protected runtime file
    ↓
authenticated decrypt into private tmpfs session
    ↓
vLLM load + Dynamo registration + serving
    ↓
stop/reap engine children + cleanup session
```

Profile `encrypted-file` không có license hoặc ràng buộc máy. Runtime khách
hàng nhận file DEK riêng của package. Ai có cả package và DEK đều có thể dùng
chúng trên máy tương thích khác. Profile `encrypted-file-license` bổ sung kiểm
tra entitlement bằng license đã ký, nhưng file key vẫn không phụ thuộc TPM.
License phần mềm không cung cấp thu hồi dựa trên phần cứng. Không bật một trong
hai profile bằng cách đổi env cho package đã ký theo profile TPM; manifest đã
ký xác định profile và các layer bắt buộc.

## 3. Thành phần và trust boundaries

```text
OFFLINE BUILD / ISSUER                       CUSTOMER NODE

Original model                              Signed runtime image
    ↓                                             ↓
Packager                                    Secure bootstrap
    ├─ validate source                           ├─ detect/verify
    ├─ fresh artifact ID + DEK                   ├─ own session
    ├─ encrypt authenticated records             └─ Rust protection core
    ├─ sign immutable manifest                         ↓
    └─ store KEK-wrapped DEK                 TPM-authorized unwrap
           ↓                                          ↓
Encrypted package                            dedicated tmpfs/model
                                                       ↓
License Issuer ── activation/reissue ──► license ──► vLLM ──► GPU
```

| Thành phần | Trách nhiệm |
|---|---|
| Packager | Validate source, sinh artifact/DEK, encrypt record, build và ký manifest |
| Offline issuer key store | Giữ encrypted private keys/KEK ngoài source và image; chỉ mở khóa trong phiên build/issue được kiểm soát |
| License Issuer | Enrollment, entitlement, package/device binding, reissue |
| TPM profile | Giữ device unwrap key; machine-bound unwrap theo policy đã duyệt |
| Secure bootstrap | Detect trước backend side effect, stage metadata, validate effective config |
| Rust protection core | Strict parsing, signature, safe path I/O, unwrap, decrypt, secret guards |
| Session owner | Volume lock, state transitions, process group, cleanup và failure recovery |
| Backend adapter | Truyền verified paths, giới hạn mode, registration và public metadata |

Rust core dùng thư viện crypto/TPM đã được review. Không tự viết AES, GCM,
signature, TPM transport hoặc secure allocator primitives.

## 4. Key hierarchy và lifecycle

### 4.1 Tách key theo mục đích

| Key | Dùng cho | Custody |
|---|---|---|
| Package signing key | Ký exact manifest bytes | Offline software issuer key store |
| License signing key | Ký exact license bytes | Offline software issuer key store |
| TPM policy authority key | Authorize TPM key-use policy | Offline software issuer key store |
| Container signing key | Ký runtime image | Release pipeline |
| Data Encryption Key (DEK) | AES-256-GCM model records | Fresh mỗi customer/model-version artifact |
| Device Unwrap Key (DUK) | Unwrap DEK trên máy được đăng ký | Private part non-exportable trong TPM |
| Issuer wrapping key (KEK) | AES-KWP bảo vệ DEK record at rest | Encrypted offline key directory |

Các key có key ID, file riêng, audit, rotation và incident procedure riêng.
Public runtime image chỉ chứa allowlisted trust roots; không chứa private
signing key, DEK, DUK hoặc issuer wrapping key. Không có HSM nghĩa là issuer
host trở thành security boundary: full-disk encryption, dedicated service
account, offline execution, owner-only permission và backup encryption là bắt
buộc.

V1 target profile được cố định như sau:

- customer runtime dùng TPM 2.0 qua resource manager `/dev/tpmrm0` và
  `tpm2-tss` ESAPI; raw `/dev/tpm0` không phải production default;
- DUK là child của owner-hierarchy storage primary theo TCG RSA-2048 SRK
  template (`restricted|decrypt`, SHA-256 Name, AES-128-CFB inner wrapper).
  Runtime lưu TPM2B_PUBLIC/TPM2B_PRIVATE blobs như activation state và tái tạo
  primary từ đúng template; copy blobs sang TPM khác không load được;
- DUK là unrestricted RSA-2048 decrypt object, `nameAlg=SHA-256`, public
  scheme/symmetric fields `NULL`, exponent mặc định 65537, với attributes
  `fixedTPM|fixedParent|sensitiveDataOrigin|adminWithPolicy|decrypt|noDA`;
  `userWithAuth` phải tắt để không có password authorization bypass;
- license bind `device_key_name` với TPM Name và
  `device_public_key_sha256 = SHA-256(marshaled TPMT_PUBLIC)`; với SHA-256
  `nameAlg`, 32 byte cuối của Name phải bằng digest này;
- DEK được wrap bằng RSA-OAEP/SHA-256 với label domain
  `model-protection-dek-v1\0`; ciphertext trong license phải đúng 256 bytes;
- DUK `authPolicy` dùng `PolicyAuthorize`. Approved policy giới hạn
  `TPM2_CC_RSA_Decrypt` và bind cpHash của exact wrapped-DEK/OAEP parameters.
  License mang signed `command_parameters_hash` để runtime gọi
  `PolicyCpHash`; TPM tự đối chiếu digest đó với command parameters thực khi
  `RSA_Decrypt` chạy.
  Policy authority là ECDSA P-256/SHA-256 riêng, signature lưu dạng raw
  P1363 `r || s` 64 bytes;
- `policyRef` là 32-byte domain digest cố định cho profile; PCR binding tắt ở
  V1. Bật PCR về sau là profile/version mới, không thay đổi ngầm;
- enrollment verify EK certificate với OEM trust store, dùng fresh
  MakeCredential/ActivateCredential để chứng minh AK possession, rồi
  `TPM2_Certify` DUK public area/Name bằng AK;
- issuer software profile dùng encrypted PKCS#8 cho hai Ed25519 package/license
  signer riêng và ECDSA P-256 TPM-policy signer; AES-256 KEK là exact 32-byte
  secret trong encrypted offline key directory. Passphrase được đưa qua
  protected file descriptor/secret file, không qua argv hoặc environment;
- key loader dùng `openat2`/bounded read, reject symlink/FIFO/device, kiểm owner,
  mode `0600`, exact format/length và key ID trước khi làm crypto. Private key
  và KEK chỉ tồn tại trong owned locked/non-dump memory, zeroize khi drop;
- issuer process gọi `mlockall(MCL_CURRENT|MCL_FUTURE)` và fail closed nếu
  `RLIMIT_MEMLOCK` không đủ; swap/core-dump/hibernation policy của offline host
  vẫn là operator control được audit, không suy ra từ cgroup desktop;
- việc volume thật sự được mã hóa là operator control và phải được
  kiểm trong runbook/audit; application không được tuyên bố có thể suy ra
  điều đó chỉ từ pathname hoặc mountpoint;
- issuer dùng AES-KWP/RFC 5649 cho issuer record và RSA-OAEP/SHA-256 với exact
  label `model-protection-dek-v1\0` để tạo ciphertext 256 byte cho TPM DUK.
  Plaintext DEK đi qua issuer process memory trong thời gian ngắn; đây là
  tradeoff được chấp nhận khi bỏ HSM và phải được ghi audit, không được log;
- packager chỉ giữ fresh 256-bit DEK trong locked process memory đủ lâu để
  stream-encrypt artifact, wrap DEK bằng software KEK rồi zeroize. Storage chỉ giữ
  KWP-wrapped DEK, key ID/version và audit metadata; không giữ plaintext DEK;
- TPM clear hoặc thay motherboard yêu cầu enrollment và license reissue. GPU
  change không đổi DUK. Offline entitlement là perpetual; không claim calendar
  expiry hoặc immediate revocation.

### 4.2 DEK scope đã chọn cho V1

Mỗi immutable artifact cho một customer và model version dùng một fresh random
256-bit DEK. Mọi rebuild, kể cả retry sau build lỗi, tạo artifact ID và DEK mới.

Tradeoff:

- leak một runtime DEK ảnh hưởng các deployment dùng cùng customer artifact;
- customer khác hoặc model version khác không bị ảnh hưởng;
- thay server chỉ rewrap cùng DEK sang TPM mới, không re-encrypt model lớn.

Packager tạo DEK trong locked Rust buffer, wrap bằng software KEK và xuất
signed issuer-record V2:

```text
exact manifest digest + artifact/customer/model identity
    → KEK ID/version + AES-KWP-wrapped DEK + issuer-record signature
```

Không lưu plaintext DEK trong package, storage, build output, environment,
command line hoặc log. Backup/restore xử lý signed wrapped record và encrypted
issuer key directory riêng biệt.

### 4.2.1 Nguyên lý mã hóa và truyền key

Runtime không nhận hoặc truyền **DEK plaintext** qua CLI, environment, Docker
image, license bundle hay network. Thứ được vận chuyển giữa các bước luôn là
ciphertext hoặc public key material. DEK chỉ xuất hiện trong bộ nhớ của
packager/issuer trong thời gian ngắn và trong `SecretDek` buffer của Rust
runtime sau khi TPM unwrap thành công.

#### A. Khi đóng gói model (issuer host)

```text
Plaintext weights
       │
       ├─ generate fresh random 256-bit DEK
       │
       ├─ AES-256-GCM(record, DEK, nonce, AAD)
       │             │
       │             └─ ciphertext records → protected package
       │
       ├─ SHA-256(ciphertext/metadata) + Ed25519(package signing key)
       │             │
       │             └─ signed manifest
       │
       └─ AES-KWP(DEK, issuer KEK)
                     │
                     └─ signed issuer-record (chỉ issuer giữ)
```

Các điểm bắt buộc:

1. DEK mới được sinh cho từng `artifact_id`; retry hoặc rebuild không tái sử
   dụng DEK/nonce.
2. Weights được mã hóa theo bounded record. GCM tag phải hợp lệ trước khi
   plaintext record được ghi vào tmpfs.
3. Packager chỉ lưu `AES-KWP(DEK, issuer KEK)` trong issuer-record. Package
   giao cho customer không chứa DEK, issuer KEK hoặc issuer-record.
4. Sau khi stream-encrypt và wrap xong, buffer chứa DEK plaintext được
   zeroize; manifest ký trên exact bytes, không re-serialize JSON.

#### B. Khi phát hành license bind với TPM

Issuer đọc issuer-record, xác thực package digest/identity và chỉ unwrap DEK
trong locked memory của issuer process. DEK sau đó được mã hóa lại bằng public
area của DUK trên TPM đích:

```text
issuer-record
  └─ AES-KWP unwrap bằng issuer KEK
                    │
                    └─ plaintext DEK (issuer memory, tạm thời)
                                      │
                                      └─ RSA-OAEP-SHA256-
                                          Encrypt(DUK public, DEK,
                                          label=model-protection-dek-v1\0)
                                                        │
                                                        └─ wrapped_dek
                                                           (256 bytes)
                                                           trong license
```

License chỉ chứa `wrapped_dek`, `device_key_name`, policy digests, package
digest và chữ ký. Private DUK không export khỏi TPM; license signing key và
issuer KEK không được giao cho customer. Vì vậy “truyền key” ở bước này thực
chất là truyền một bản DEK đã được mã hóa cho đúng TPM, không phải truyền DEK
plaintext.

#### C. Khi giao package cho customer

Customer nhận các artifact sau:

```text
protected-package/       ciphertext weights + signed manifest + public metadata
license-bundle/          signed license + RSA-OAEP wrapped_dek
runtime trust roots/     public Ed25519/ECDSA keys
```

Không giao:

```text
plaintext model
plaintext DEK
issuer-record.json
issuer KEK
package/license private signing keys
TPM private DUK
```

V1 offline không có key-release network call. License là file ciphertext đã ký,
được mount read-only vào runtime. Nếu sau này có online profile, chỉ thêm một
kênh authenticated/attested để release hoặc rewrap key; không hạ cấp về file
key fallback.

#### D. Khi runtime khởi động

```text
license + package
        │
        ├─ verify exact signatures, schema, digest và device binding
        │
        ├─ verify effective vLLM configuration
        │
        └─ TPM2 RSA_Decrypt(wrapped_dek, DUK private, PolicyAuthorize)
                                      │
                                      └─ DEK plaintext trả về qua ESAPI
                                         vào Rust SecretDek (opaque/locked)
```

TPM chỉ thực hiện unwrap; TPM không đọc hoặc giải mã toàn bộ model. Sau khi
nhận `SecretDek`, Rust decryptor:

1. đọc từng ciphertext record từ package;
2. kiểm tra nonce/counter, AAD và GCM authentication tag;
3. ghi plaintext đã authenticate vào file `.partial` trong
   `/run/<validated-DYN_NAMESPACE>-models/<session-id>`;
4. kiểm tra đủ size/hash rồi atomic rename thành tên weights cuối;
5. zeroize record buffer và `SecretDek` ngay khi không còn cần.

`SecretDek` không được chuyển sang Python object, argv, environment, log,
license file hoặc filesystem thường. ESAPI/TCTI là transport nội bộ giữa
runtime và `/dev/tpmrm0`; đây không phải network transport và không làm private
DUK rời TPM.

#### E. Vì sao copy package sang máy khác không chạy

```text
Server A: wrapped_dek ── DUK-A private trong TPM-A ── unwrap OK
Server B: wrapped_dek ── không có DUK-A ──────────── unwrap FAIL
```

License còn bind `device_key_name`/public digest với DUK-A. Do đó copy cả
package và license sang Server B sẽ thất bại ở bước signature/device binding
hoặc TPM `RSA_Decrypt`; không có DEK để giải mã weights. Copy riêng package
cũng vô dụng vì package chỉ chứa ciphertext.

#### F. Những cách truyền key bị cấm

```text
MODEL_KEY=...                         # không dùng environment
--model-key <plaintext>               # không dùng CLI
/models/package/dek.bin               # không lưu DEK plaintext
COPY issuer-key /app/                 # không đưa private key vào image
license.json chứa plaintext `dek`    # chỉ cho phép `wrapped_dek`
```

Các quy tắc này áp dụng cho cả local Docker và Kubernetes. Khác biệt chỉ là
secret/license mount và tmpfs volume; nguyên lý unwrap và custody của DEK
không thay đổi.

### 4.2.2 Luồng mã hóa model từng bước

Phần này chỉ mô tả **đóng gói model plaintext thành encrypted package**. TPM,
license và vLLM không tham gia vào việc mã hóa ciphertext ban đầu; chúng chỉ
được dùng ở bước phát hành license và runtime unwrap sau đó.

#### Bước 0 — Chuẩn bị đầu vào và trust boundary

Thực hiện trên issuer/build host đã được kiểm soát, tốt nhất là volume đã mã
hóa và không kết nối mạng trong lúc dùng private key.

| Thành phần | Lấy từ đâu | Dùng để làm gì | Có giao cho customer không? |
|---|---|---|---|
| Plaintext model | Thư mục model HF/safetensors do team model cung cấp | Nguồn đọc một lần để tạo ciphertext và public metadata | Không |
| Package signing key (Ed25519 private) | Encrypted offline key directory | Ký manifest và issuer-record | Không |
| Package signing public key | Sinh từ private key, phát hành qua runtime trust bundle | Verify package ở runtime | Có |
| Issuer KEK (AES-256, 32 byte) | Encrypted offline key directory | AES-KWP-wrap DEK khi lưu issuer-record | Không |
| Passphrase files | Secret file owner-only/protected FD | Mở private key và KEK | Không |
| Packager binary | Build từ source revision đã review | Validate, encrypt, hash, sign và wrap | Không cần giao |
| Output directory | Encrypted storage có đủ dung lượng | Lưu package và issuer-record | Chỉ package được giao |

TPM/DUK **chưa cần** ở bước này. DUK public chỉ được dùng về sau khi issuer
phát hành license cho một máy cụ thể.

#### Bước 1 — Validate model source

Packager nhận `--source /absolute/path/to/model` và không sửa source. Nó:

1. kiểm tra source là directory local, không phải URI từ xa;
2. phân loại file thành weights safetensors và public metadata theo allowlist;
3. reject symlink, FIFO, device, path traversal, file ngoài inventory hoặc
   kích thước vượt signed bounds;
4. đọc metadata/config/tokenizer cần thiết và tính size/hash trên bytes thực;
5. fail closed nếu model không phù hợp contract V1.

Đầu ra của bước này là một inventory nội bộ. Chưa có DEK và chưa có file
ciphertext nào được publish.

#### Bước 2 — Sinh DEK mới

Packager dùng CSPRNG của Rust để sinh một **DEK 256-bit (32 byte)** cho đúng
`artifact_id` hiện tại:

```text
artifact_id (random, unique)
        │
        └─ CSPRNG → DEK (32 bytes)
```

DEK thuộc phạm vi một customer/model-version artifact. Mỗi lần build hoặc retry
phải sinh DEK và `artifact_id` mới; không lấy DEK từ config, environment, CLI,
model directory hay license cũ. DEK nằm trong locked/owned memory, không log và
không được ghi ra file plaintext.

#### Bước 3 — Stream-encrypt weights

Mỗi safetensors weight file được đọc theo bounded record, không nạp cả model
vào một Python buffer:

```text
plaintext shard
   ├─ read bounded chunk
   ├─ create record header (file_id, index, counter, length)
   ├─ derive nonce = nonce_prefix || global_counter
   ├─ build exact AAD from artifact/file/record fields
   ├─ AES-256-GCM encrypt bằng DEK
   └─ ghi ciphertext + 16-byte tag vào .partial
```

Quy tắc của record:

- `nonce_prefix` 32-bit được sinh mới cho artifact; `global_counter` 64-bit
  tăng liên tục và không được lặp;
- AAD xác thực identity và thứ tự record, nên đổi file ID, index, counter hoặc
  plaintext length sẽ làm tag sai;
- chỉ record có GCM tag hợp lệ mới được coi là ciphertext hoàn chỉnh;
- output luôn ghi vào `weights/<name>.safetensors.protected.partial`, sau khi
  đủ record và kiểm tra kích thước mới atomic rename thành tên cuối;
- source plaintext chỉ được đọc, không bị sửa hoặc ghi ngược vào package.

Nếu mất điện, hết dung lượng hoặc tag/record lỗi, xóa toàn bộ output session và
không để file `.partial` được dùng như model hợp lệ.

#### Bước 4 — Hash ciphertext và public metadata

Sau khi từng file hoàn thành, packager tính và giữ trong inventory:

```text
container_sha256   = SHA-256(toàn bộ file .protected)
plaintext_sha256   = SHA-256(bytes plaintext đã đọc)
container_size     = kích thước ciphertext + header/tag
plaintext_size     = kích thước source
record_count       = số record đã tạo
```

Public files như `config.json`, tokenizer và safetensors index được copy sang
`public/` theo allowlist, rồi hash exact bytes. Chúng không được mã hóa trong
V1 vì bootstrap/vLLM cần đọc trước khi release DEK; integrity của chúng vẫn
được manifest ký bảo vệ.

#### Bước 5 — Tạo manifest bất biến

Packager tạo `model.protection.json` từ inventory và metadata do caller cung
cấp:

```text
artifact_id
customer_scope_id
model_id / model_version
encryption profile
protected file inventory + hashes + sizes
public file mapping + hashes
runtime/load-format requirements
```

Manifest không chứa DEK, KEK, passphrase hay private key. JSON được serialize
một lần thành exact UTF-8 bytes; runtime về sau phải verify đúng các bytes này,
không parse rồi re-serialize để tính chữ ký.

#### Bước 6 — Ký manifest

Packager đọc Ed25519 private package-signing key từ encrypted offline key
directory bằng passphrase file. Nó ký:

```text
model-protection-manifest-signature-v1\0 || manifest_bytes
```

và ghi chữ ký vào `model.protection.sig`. Public key tương ứng được đưa vào
runtime trust bundle. Nếu ký lỗi hoặc private key không đúng key ID, build dừng
và không phát hành package.

#### Bước 7 — Wrap DEK để issuer lưu trữ

DEK plaintext vẫn chỉ nằm trong locked memory của packager. Trước khi kết thúc,
packager đọc AES-256 issuer KEK từ key directory và tạo:

```text
wrapped_dek_for_issuer = AES-KWP(issuer_KEK, DEK)
```

Issuer-record V2 chứa tối thiểu:

```text
exact manifest/package digest
artifact/customer/model identity
KEK key ID + version
wrapped_dek_for_issuer
issuer-record signature
```

Issuer-record dùng cho issuer cấp license, **không nằm trong package giao
customer**. Nếu wrap hoặc ký association record không thành công, build fail
closed và DEK được zeroize.

#### Bước 8 — Finalize và zeroize

Packager kiểm tra lại package layout, manifest signature, file count/size/hash,
record range và không có plaintext weight ngoài source directory. Sau đó:

1. flush/close file handles;
2. atomic publish package directory;
3. zeroize DEK, nonce state và temporary plaintext buffers;
4. chỉ giữ ciphertext package và issuer-record đã wrap.

Kết quả hợp lệ:

```text
OUTPUT/
├── package/
│   ├── model.protection.json
│   ├── model.protection.sig
│   ├── public/*
│   └── weights/*.safetensors.protected
└── issuer-record.json       # internal issuer only
```

#### Bước 9 — Handoff sang issuer/license (không phải package encryption)

Sau khi package hoàn tất, issuer mới dùng `issuer-record.json` và public DUK
của máy đích để tạo:

```text
DEK ← AES-KWP unwrap bằng issuer KEK (issuer memory)
wrapped_dek ← RSA-OAEP-SHA256-Encrypt(DUK public, DEK)
license ← signed package/device entitlement + wrapped_dek
```

Bước này không thay đổi ciphertext hoặc manifest. Chỉ `wrapped_dek` được đưa
vào license; DEK plaintext và issuer-record không rời issuer host.

#### Bước 10 — Artifact được giao

Customer chỉ nhận:

```text
package/                  # ciphertext + signed manifest + public metadata
license/model.protection.license.json
license/model.protection.license.sig
trust/package-public.raw
trust/license-public.raw
runtime.json
```

Không nhận plaintext model, DEK, issuer KEK, issuer-record, passphrase hoặc
private signing key. Vì license chứa DEK đã mã hóa cho DUK cụ thể, copy package
không kèm TPM tương ứng không thể chạy; copy cả package và license sang TPM
khác vẫn thất bại ở device binding/TPM unwrap.

### 4.2.3 Giải thích thuật ngữ trong luồng mã hóa

Đọc nhanh luồng chính:

```text
model plaintext
  → DEK
  → AES-256-GCM
  → ciphertext package
  → manifest hash/signature
  → DEK được AES-KWP wrap để issuer lưu
  → issuer RSA-OAEP-wrap DEK tới TPM DUK khi cấp license
```

| Thuật ngữ | Giải thích dễ hiểu | Vai trò trong hệ thống |
|---|---|---|
| **Plaintext** | Dữ liệu ở dạng đọc được, ví dụ file `.safetensors` gốc | Chỉ tồn tại ở model source và buffer tạm của packager; không được giao customer |
| **Ciphertext** | Dữ liệu sau khi mã hóa, không thể đọc nếu thiếu key | Là nội dung các file `*.safetensors.protected` trong package |
| **Key** | Chuỗi byte bí mật dùng bởi thuật toán mã hóa; không phải tên file hay password | Không được hard-code, log hoặc truyền qua CLI/environment |
| **DEK (Data Encryption Key)** | Khóa dữ liệu đối xứng 256-bit, sinh ngẫu nhiên cho một artifact | Dùng trực tiếp với AES-256-GCM để mã hóa weights; được zeroize sau khi dùng |
| **KEK (Key Encryption Key)** | Khóa dùng để mã hóa một khóa khác | Issuer KEK dùng AES-KWP để bảo vệ DEK trong issuer-record; không mã hóa weights |
| **DUK (Device Unwrap Key)** | Khóa giải-wrap gắn với một máy, có private part nằm trong TPM | TPM dùng DUK private để mở `wrapped_dek`; private DUK không export được |
| **TPM 2.0** | Chip/module bảo mật phần cứng có vùng lưu private key và policy | Chứng minh DEK chỉ được mở trên máy đã đăng ký; không trực tiếp giải mã model |
| **TPM Name / `device_key_name`** | Định danh bất biến được tính từ public area của TPM object | License bind định danh này để copy sang TPM khác bị từ chối |
| **CSPRNG** | Bộ sinh số ngẫu nhiên an toàn bằng entropy hệ điều hành | Sinh `artifact_id`, DEK và nonce prefix; không dùng `random()` thông thường |
| **AES-256-GCM** | Mã hóa đối xứng có cả confidentiality và integrity | Mã hóa từng record weights và tạo authentication tag |
| **Nonce** | Giá trị không được lặp lại với cùng DEK; không phải secret | Ghép `nonce_prefix` với counter để mỗi record có nonce duy nhất |
| **Counter** | Số thứ tự tăng dần của record | Chống lặp, reorder và nonce reuse; overflow phải làm build fail |
| **AAD (Additional Authenticated Data)** | Metadata được xác thực nhưng không mã hóa | Bind artifact/file/record identity vào GCM tag |
| **GCM tag / authentication tag** | Tem kiểm tra chứng minh ciphertext và AAD không bị sửa | Tag sai thì không được ghi plaintext record vào tmpfs |
| **Record / chunk** | Một phần nhỏ của shard, được mã hóa độc lập | Giảm peak RAM và cho phép verify trước từng lần ghi |
| **SHA-256 / hash** | Hàm tạo dấu vân tay từ bytes; không phải encryption | Kiểm tra file/package có đúng bytes đã ký hay không |
| **Manifest** | Bản kê có chữ ký của artifact, file, hash, size và policy | Runtime dùng để biết file nào hợp lệ và phải reject file thừa/sai |
| **Exact bytes** | Đúng chuỗi byte đã ký, không parse rồi serialize lại | Tránh thay đổi khoảng trắng/encoding làm lệch chữ ký |
| **Ed25519 signature** | Chữ ký bất đối xứng: private key ký, public key verify | Ký manifest/package và license; customer chỉ có public key |
| **Public key** | Phần có thể công khai để kiểm tra chữ ký hoặc mã hóa cho recipient | Runtime nhận trust root và issuer dùng public DUK để wrap DEK |
| **Private key** | Phần bí mật dùng ký hoặc unwrap | Private signing key ở issuer; private DUK ở TPM; không giao customer |
| **Issuer** | Máy/quy trình phát hành artifact và license của vendor | Giữ private key/KEK, tạo package và rewrap DEK tới DUK đích |
| **Issuer-record** | Hồ sơ nội bộ chứa DEK đã AES-KWP-wrap và association metadata | Dùng để cấp/reissue license; tuyệt đối không ship cùng package |
| **Wrap** | Mã hóa một key bằng key khác để lưu/truyền an toàn | `AES-KWP(DEK, KEK)` hoặc `RSA-OAEP(DUK public, DEK)` |
| **Unwrap** | Mở key đã wrap bằng đúng key bảo vệ nó | Issuer unwrap bằng KEK; TPM unwrap bằng DUK private |
| **RSA-OAEP/SHA-256** | Cơ chế mã hóa bất đối xứng cho một plaintext nhỏ như DEK | Wrap DEK tới TPM DUK; V1 yêu cầu ciphertext đúng 256 bytes |
| **OAEP label** | Chuỗi domain dùng khi wrap/unwrap để tách mục đích sử dụng | V1 cố định `model-protection-dek-v1\0`; label khác sẽ fail |
| **SecretDek** | Handle/kiểu opaque chỉ Rust core tạo sau TPM unwrap | Ngăn caller tự truyền raw DEK; chỉ decryptor nội bộ được tiêu thụ |
| **tmpfs** | Filesystem nằm chủ yếu trong RAM thay vì disk thường | Nơi ghi plaintext weights tạm thời trước khi vLLM load |
| **Atomic rename** | Ghi file `.partial` rồi đổi tên trong một thao tác nguyên tử | vLLM không bao giờ nhìn thấy file decrypt dở |
| **Zeroize** | Ghi đè/xóa buffer bí mật trước khi giải phóng | Giảm thời gian DEK/plaintext tồn tại trong process memory |
| **Trust root** | Public key được runtime tin cậy từ trước | Verify package/license; mỗi trust domain dùng key riêng |
| **PolicyAuthorize** | TPM policy cho phép authority ký duyệt một policy digest | Giới hạn DUK chỉ được dùng cho đúng command/DEK đã được license duyệt |

Điểm cần nhớ: **DEK là khóa mã hóa model; KEK chỉ bảo vệ DEK khi issuer lưu;
DUK là khóa phần cứng dùng để mở DEK trên đúng máy**. Ba khóa này có mục đích,
nơi lưu và vòng đời khác nhau, không được dùng lẫn nhau.

### 4.3 Offline TPM semantics

Enrollment tạo TPM Attestation Key (AK) và DUK. Issuer xác thực EK/AK theo
deployment trust policy, certify đúng DUK và verify fresh nonce trong quote.
Activation transcript bind:

- protocol version;
- issuer challenge và expiry của challenge;
- package digest;
- license/customer/product IDs;
- AK identity và certified DUK Name/public area;
- requested hardware policy.

Issuer rewrap DEK tới DUK và phát license có chữ ký. TPM policy authority data
bind ít nhất package digest, license ID, DUK Name và entitlement generation.
TPM object không có alternate password/bypass authorization path.

Profile ở mục 4.1 cố định object attributes, command restriction, algorithms,
`policyRef`, PCR use và reset behavior. Command-level golden vector đã sinh
public-area/Name, `command_parameters_hash`, approved policy và
`PolicyAuthorize` signature trên `swtpm`; ESAPI feature cũng build/link với
`tpm2-tss`. Đây chưa thay physical TPM review. `PolicyAuthorize` chỉ authorize policy digest; nó không tự
parse JSON hoặc tự enforce GPU/license claims.

Offline V1 dùng perpetual entitlement. GPU inventory là signed runtime policy
và support signal; nó không phải secret hay hardware root of trust. Nếu root
operator patch local check, điều đó nằm ngoài V1 threat model.

### 4.4 Hardware replacement

- GPU thay đổi trong policy không đổi DUK; runtime ghi sanitized audit event.
- Thay TPM/motherboard hoặc TPM clear tạo device identity mới.
- Khách gửi activation request mới; issuer rewrap DEK tới DUK mới và tăng
  entitlement generation.
- Issuer deny generation cũ khi phát hành/reissue; node offline cũ vẫn dùng
  perpetual license cho tới khi artifact hoặc deployment bị thu hồi vật lý.
- Package ciphertext và manifest không thay đổi khi reissue license.

### 4.5 Online profile sau V1

Nếu sản phẩm cần calendar expiry hoặc revocation nhanh, dùng online attested
key service:

- verifier phát one-time challenge;
- evidence bind challenge, package, license, device và recipient session key;
- service release DEK qua authenticated channel tới đúng recipient;
- serving lease tách khỏi DEK, có renewal/deadline;
- hết lease: ngừng admission, bounded drain, dừng process group, cleanup;
- network failure dùng signed bounded grace policy, không file-key fallback.

Không gửi lại DEK khi chỉ renew serving lease. Revocation vẫn không thu hồi
weights đã bị attacker extract trước đó.

## 5. Artifact và license format

### 5.1 Reserved package layout

```text
protected-model-package/
├── model.protection.json
├── model.protection.sig
├── public/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── preprocessor_config.json
│   └── model.safetensors.index.json
└── weights/
    ├── model-00001-of-00004.safetensors.protected
    └── model-00002-of-00004.safetensors.protected

license-bundle/
├── model.protection.license.json
└── model.protection.license.sig
```

Secure detector chỉ kiểm tra hai reserved control names tại package root:

- `model.protection.json`;
- `model.protection.sig`;

Không dùng `.bin`, `manifest.json`, `.sig` generic hoặc quét suffix của toàn bộ
directory làm marker. Presence của một trong hai root marker nhưng package
không đầy đủ là `SECURE_INVALID`; khi cả hai marker vắng, input đi theo plain
path. Protected container vẫn phải là direct child
`weights/<name>.safetensors.protected` và chỉ được đọc sau khi exact root
manifest đã verify. Contract này giữ plain detection O(1), tránh giới hạn số
file của model thường, và không tạo đường decrypt cho một ciphertext bị xóa cả
hai control marker. Remote secure URI không được hỗ trợ trong V1.

### 5.2 Manifest structure

Manifest ký domain separator cộng exact UTF-8 bytes, reject duplicate JSON keys, unknown critical
fields, invalid Unicode, values ngoài bounds và unsupported version. V1 pin
Ed25519 cho package/license signatures và AES-256-GCM cho records.

Schema logic:

```json
{
  "format": "secure-model-package",
  "format_version": 1,
  "artifact_id": "128-bit-random-id",
  "customer_scope_id": "opaque-id",
  "model": {
    "model_id": "model-family",
    "model_version": "1.0.0",
    "framework": "safetensors"
  },
  "encryption": {
    "algorithm": "AES-256-GCM",
    "nonce_prefix": "4-byte-hex",
    "tag_bits": 128,
    "record_plaintext_limit": 16777216
  },
  "protected_files": [
    {
      "file_id": 1,
      "container_path": "weights/model-00001-of-00004.safetensors.protected",
      "output_path": "model-00001-of-00004.safetensors",
      "container_size": 123456789,
      "container_sha256": "...",
      "plaintext_size": 123450000,
      "plaintext_sha256": "...",
      "record_count": 8,
      "first_global_record_counter": 0
    }
  ],
  "public_files": [
    {
      "source_path": "public/config.json",
      "output_path": "config.json",
      "size": 1234,
      "sha256": "...",
      "publish_to_model_card": true
    }
  ],
  "runtime": {
    "minimum_runtime_version": "approved-version",
    "required_load_format": "safetensors"
  }
}
```

Một `output_path` xuất hiện đúng một lần trong toàn manifest. Nhiều records nằm
trong một protected container/file entry; chúng không tạo duplicate output.
Plaintext full-file size/hash và container full-file size/hash đều bắt buộc.

Package digest được định nghĩa là SHA-256 của domain separator và exact manifest
bytes; signature envelope không nằm trong digest. License bind exact package
digest. Cách này cho phép rotate/re-sign package key mà không đổi artifact
identity. Không dùng package digest làm AEAD AAD vì digest phụ thuộc vào
ciphertext hashes.

V1 foundation áp dụng các bound sau: manifest tối đa 4 MiB, tổng file tối đa
4096, path ASCII tương đối tối đa 1024 bytes với ký tự `[A-Za-z0-9/._-]`, một
component tối đa 240 bytes để dành chỗ cho `.partial`, một file tối đa 1 TiB,
tối đa 1,048,576 records/artifact và một record plaintext tối đa 64 MiB.
Producer có thể chọn limit nhỏ hơn trong signed manifest. Signature bao phủ
exact bytes đã giao; runtime không parse rồi re-serialize để verify.
Digest dùng exact byte domain
`model-protection-manifest-v1\0 || manifest_bytes`. Chữ ký package dùng domain
riêng `model-protection-manifest-signature-v1\0 || manifest_bytes`; chữ ký
license dùng `model-protection-license-signature-v1\0 || license_bytes` để
không thể chuyển chữ ký giữa hai loại artifact.

### 5.3 Authenticated record format

Mỗi container là chuỗi record không có unauthenticated trailer. Header dài 36
bytes; integer dùng unsigned big-endian:

```text
offset  size  field
0       8     magic = "MPROTV1\0"
8       2     record_version = 1
10      2     reserved = 0
12      4     file_id
16      4     record_index
20      8     global_record_counter
28      4     plaintext_length
32      4     ciphertext_length (= plaintext_length trong V1)
36      N     ciphertext
36+N    16    GCM tag
```

Nonce 96-bit:

```text
nonce = artifact nonce_prefix (32-bit) || global_record_counter (64-bit big-endian)
```

Counter duy nhất và tăng liên tục trên toàn artifact. Build dừng nếu counter
overflow, record thiếu/lặp/reorder hoặc output vượt signed limits. Fresh DEK
mỗi artifact build ngăn retry tái sử dụng key/counter domain.

AAD có exact binary encoding:

```text
"model-protection-record-v1\0" ||
record_version:u16 || artifact_id:16 bytes || file_id:u32 ||
record_index:u32 || global_record_counter:u64 || plaintext_length:u32
```

Manifest yêu cầu `record_count = ceil(plaintext_size /
record_plaintext_limit)`, `container_size = plaintext_size + record_count * 52`
và các counter range không overlap. Mọi record trừ record cuối phải đúng signed
record limit; record cuối phải khớp phần bytes còn lại. Parser reject
unknown field, duplicate JSON field, uppercase/invalid hex, path traversal,
reserved output name, size/count overflow và appended/truncated record data.

Loader đọc một bounded record, decrypt vào fixed-capacity private buffer, hoàn
tất tag verification rồi mới ghi authenticated plaintext vào `.partial` trên
tmpfs. Owned plaintext/tag buffers được zeroize trước reuse/drop. Loader đồng
thời tính container hash trên bytes thực đọc và plaintext full-file hash.
Materializer chỉ nhận opaque `SecretDek` theo ownership; constructor production
chỉ tồn tại trong module ESAPI khi build bật feature `tpm2`. Bằng chứng zeroization cho
key schedule nội bộ của crypto library vẫn là production gate.

Whole-shard GCM có thể an toàn nếu unauthenticated staging hoàn toàn không thể
được dùng trước final tag. V1 chọn independent records để duy trì invariant đơn
giản hơn: chỉ authenticated plaintext được ghi vào tmpfs.

### 5.4 License structure

Offline V1 không có `expires_at` giả tạo:

```json
{
  "format": "model-protection-license",
  "format_version": 1,
  "license_id": "license-opaque-id",
  "customer_scope_id": "customer-a",
  "artifact_id": "32-lowercase-hex-characters",
  "manifest_sha256": "64-lowercase-hex-characters",
  "model_id": "model-id",
  "model_version": "model-version",
  "entitlement": {
    "mode": "offline-perpetual",
    "generation": 1
  },
  "recipient": {
    "kind": "tpm2",
    "profile": "tpm2-offline-rsa2048-oaep-sha256-v1",
    "device_key_name": "000b-followed-by-64-lowercase-hex-characters",
    "device_public_key_sha256": "64-lowercase-hex-characters",
    "command_parameters_hash": "64-lowercase-hex-characters",
    "approved_policy_digest": "64-lowercase-hex-characters",
    "policy_ref": "30d14d0ec65c3ccbf79f0cddddc9db09a3676a43910fde63e3889f65a9a371a6",
    "policy_authority_key_id": "tpm-policy-v1",
    "policy_signature_algorithm": "ecdsa-p256-sha256-p1363",
    "policy_signature": "base64-exactly-64-bytes"
  },
  "wrapped_dek": "base64-exactly-256-bytes"
}
```

License signature envelope có algorithm/key ID/signature và được verify bằng
license trust root riêng. TPM policy authority key có thể dùng algorithm khác
phù hợp TPM; hai chữ ký không được nhập làm một.

### 5.5 Public metadata policy

Signed metadata có integrity nhưng không có confidentiality. V1 coi config,
tokenizer, chat template và processor assets đã khai báo là public. Foundation
V1 chỉ mã hóa safetensors weights; confidential metadata chưa được support vì
bootstrap cần đọc config/tokenizer trước key release. Bổ sung loại dữ liệu này
cần một schema/bootstrap contract riêng.

Foundation V1 chỉ nhận direct-root metadata với mapping chính xác
`public/<output_path>` và allowlist sau:

- `config.json`, `generation_config.json`;
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
  `added_tokens.json`;
- `vocab.json`, `vocab.txt`, `merges.txt`, `tokenizer.model`,
  `sentencepiece.bpe.model`, `spiece.model`;
- `preprocessor_config.json`, `processor_config.json`;
- `chat_template.json`, `chat_template.jinja`;
- `model.safetensors.index.json`.

Public `.safetensors`, `.bin`, Python và control/license files bị reject vì
không nằm trong allowlist. Safetensors index tối đa 4 MiB, chỉ được reference
declared protected outputs và phải reference đủ mọi output. Nhiều protected
weight files bắt buộc có index. Mở rộng inventory cho model/backend khác cần
compatibility evidence và cập nhật đồng thời packager/loader contract.

Runtime model directory chỉ chứa:

- declared model weights;
- declared metadata có `publish_to_model_card=true` hoặc assets cần vLLM load.

License, wrapped key, owner lock, audit và session state nằm ngoài model
directory để Dynamo `LocalModel` không harvest/publish chúng.

## 6. Filesystem và tmpfs contract

### 6.1 Runtime layout

Resolved root:

```text
/run/<validated-effective-namespace>-models/
├── .owner.lock
└── <random-session-id>/
    ├── config.json            # whole UUID directory is passed to vLLM
    ├── tokenizer.json
    └── *.safetensors
```

Mỗi Pod có dedicated mount và một owner. Mount root được provision cho dedicated
worker UID; session directories là `0700`, model files `0600`, `umask 077`.
Namespace/session names không phải access control.

### 6.2 Effective namespace

Path dùng namespace cuối cùng sau precedence hiện có:

- CLI/config namespace;
- `DYN_NAMESPACE` fallback;
- `DYN_NAMESPACE_WORKER_SUFFIX`;
- endpoint override nếu cấu hình đó thay namespace runtime.

Runtime không đọc lại riêng raw environment sau khi config đã resolve. Secure
validator reject empty, separator, `.`/`..`, ambiguous Unicode/normalization,
và value ngoài contract: 1-128 ASCII bytes, ký tự đầu/cuối là alphanumeric,
ký tự giữa chỉ thuộc `[A-Za-z0-9._-]`. Operator và runtime dùng cùng conformance
vectors. Kubernetes renderer phải tạo concrete mount path;
`volumeMount.mountPath` không expand environment variable.

Nếu mount không tồn tại hoặc mismatch, secure model fail. Không `mkdir` fallback
trên root filesystem. Plain model không yêu cầu mount.

Kubernetes V1 dùng một concrete namespace và override worker suffix thành chuỗi
rỗng để operator rollout hash không đổi storage namespace. Ví dụ namespace
`protected` luôn map tới `/run/protected-models`; frontend/backend phải dùng
cùng `DYN_NAMESPACE`.

### 6.3 Race-safe input/output

Package source được mount read-only và không writable bởi worker UID. Rust core
mở package/session root thành directory FD. Trên supported Linux dùng
`openat2` với:

- `RESOLVE_BENEATH`;
- `RESOLVE_NO_SYMLINKS`;
- `RESOLVE_NO_MAGICLINKS`;
- `RESOLVE_NO_XDEV` dưới từng trusted root;
- `O_CLOEXEC` và safe create flags.

Fallback chỉ được support nếu duyệt từng path component bằng directory FD với
tương đương security semantics; không fallback sang string concatenate/open.
Paths phải relative, canonical theo format, unique và không chứa NUL/traversal.
Reject symlink, hard-link count bất thường, FIFO, socket, device, sparse/size
abuse, nested mount và reserved control-name collision.

Public metadata được bounded-copy vào private staging, hash chính bytes vừa
copy, rồi atomic publish khi khớp manifest. Ciphertext AEAD và full-container
hash kiểm tra bytes thực đọc khi decrypt. Source inode bị sửa giữa các bước làm
verification fail; không tin preflight hash hoặc retained FD là snapshot.

Output root phải rỗng trước materialization. Output tạo `<name>.partial` bằng
no-overwrite exclusive create và RAII guard xóa partial trên read/write/sync,
verification hoặc rename failure. Sau record, size/hash và exact EOF checks:
close/flush, atomic no-replace rename. Session chỉ được trả về sau khi tất cả
file đã publish thành công; V1 không dùng ready-marker file.

### 6.4 Mount và persistence profile

Dedicated tmpfs không dùng chung `/dev/shm`. Docker mount dùng
`rw,noexec,nosuid,nodev,noswap,size=<limit>,mode=0700` nếu kernel hỗ trợ
`noswap`. Kubernetes memory-backed `emptyDir` cần explicit `sizeLimit`, memory
request/limit và ownership setup; actual filesystem/mount must be verified.
Protected-worker `/tmp` cùng HF/Triton/JIT cache paths dùng một bounded
memory-backed volume riêng để model-derived cache không rơi xuống node disk.

Strict V1 profile yêu cầu:

- host swap disabled; tmpfs `noswap` thêm defense-in-depth khi có;
- hibernation disabled;
- core dump, kdump, CRIU và crash collection cho workload bị chặn;
- không backup/snapshot secure volume;
- read-only root filesystem và allowlisted writable mounts;
- không privileged container, host PID hoặc `CAP_SYS_PTRACE`/`SYS_ADMIN`;
- controls áp dụng cho loader và mọi engine child process.

Encrypted swap/dump không nằm trong V1 contract. `noexec` không chặn Python
interpret code, nên secure mode vẫn reject `trust_remote_code`.

## 7. Resource model

Tách hai budget:

```text
tmpfs_bytes = total plaintext files + public metadata + filesystem overhead

process_extra_peak = AEAD buffers + hash/cipher state + vLLM load staging +
                     tokenizer/config parsing + runtime overhead

memory_peak = baseline RSS + resident tmpfs pages + process_extra_peak + margin
```

Preflight kiểm tra:

- signed size/count limits trước allocation;
- tmpfs blocks/inodes và `sizeLimit`;
- host `MemAvailable`;
- effective cgroup v2 memory limits ở leaf và ancestors;
- swap policy cho tất cả process cgroups;
- concurrency: V1 chỉ một decrypt/load owner trên volume.

Preflight không reserve memory. Operator phải cấp Kubernetes request/limit theo
measured peak; runtime vẫn xử lý ENOSPC/ENOMEM/OOM/cancellation. Không cộng AEAD
buffer vào `tmpfs_bytes` và không mặc định cộng mmap pages hai lần. Mỗi
model/load strategy có measured peak và margin được ghi trong compatibility
matrix.

## 8. Secure bootstrap trong Dynamo

### 8.1 Vì sao cần bootstrap hai pha

`parse_args()` hiện dựng `AsyncEngineArgs`; vLLM có thể load plugins trong
`EngineArgs.__post_init__`. `create_engine_config()` có thể normalize hoặc thay
model/tokenizer khi cấu hình speculative decoding. Vì vậy resolver đặt ngay
trước `AsyncLLM.from_vllm_config()` là quá muộn.

### 8.2 Phase A — structural bootstrap

Một pre-parser nhỏ chạy trước `parse_args()` và không dựng vLLM EngineArgs:

1. Đọc `--model`, protection/license inputs, served identity và structural
   flags cần phân loại.
2. Detect plain/secure bằng local reserved markers với size/path bounds.
3. Plain trả original argv để luồng hiện có tiếp tục.
4. Secure reject remote source và raw forbidden modes có thể gây side effect.
5. Verify manifest signature/schema và license/package/device binding.
6. Resolve effective namespace bằng cùng config precedence contract.
7. Acquire volume owner lock; recover stale session an toàn; verify tmpfs và
   resource baseline.
8. Tạo UUID session directory; bounded-copy/hash declared public metadata
   trực tiếp vào directory này.
9. Rewrite nội bộ model/tokenizer/config arguments sang verified session model
   path; giữ original source và served identity riêng.
10. Áp dụng approved plugin/environment policy trước vLLM EngineArgs creation.

Pre-parser hỗ trợ exact `--model value` và `--model=value` syntax; duplicate hoặc
ambiguous protected-model argument bị reject. Nó không diễn giải toàn bộ vLLM
CLI và không chạy model-dependent code.

### 8.3 Phase B — effective config gate

Gọi Dynamo `parse_args()` với internal rewritten argv, sau đó validate effective
config object:

- engine model, model_weights, tokenizer và config path đều nằm trong
  UUID session directory hoặc đúng signed allowlist;
- served model name bằng explicit user value hoặc original model identity;
- load format và worker/executor topology nằm trong compatibility allowlist;
- plugin, loader, template, LoRA, speculative model và weight transfer không
  đưa source ngoài package;
- registration dùng exact verified model path, không fallback original source;
- `DYN_NAMESPACE` path khớp effective resolved namespace/mount.

Validation failure cleanup session trước DEK release.

### 8.4 V1 backend allowlist

| Capability | V1 |
|---|---|
| Local safetensors, explicit safetensors loader | Implement; còn thiếu GPU/vLLM 0.28 acceptance |
| Tensor parallel trong cùng Pod/node | Reject trong V1 product slice; chỉ `TP=1` |
| Data parallel/Ray/multi-node engine | Reject |
| Disaggregated Pods | Phase sau; mỗi Pod/node phải independently authorize/decrypt |
| Snapshot/CRIU | Reject |
| `trust_remote_code` | Reject |
| Dynamic LoRA | Reject |
| RL/update weights from disk/distributed/tensor | Reject |
| Speculative external model | Reject |
| ModelExpress, GMS, RunAI or remote loader | Reject |
| `np_cache`, dummy/custom loader, external worker class | Reject |
| External tokenizer/config/chat template | Reject; include and sign trong package |
| Headless và multi-process embedding pool | Reject cho tới lifecycle review |

Không âm thầm sửa forbidden flag rồi chạy. Trả stable configuration error trước
key release.

### 8.5 Key release và materialization

Sau effective config gate:

1. Khởi tạo RAII secret guard trước khi nhận secret bytes.
2. Verify license lần cuối trong Rust core và match artifact/device/policy.
3. Gọi TPM authorization/unwrap; DEK chỉ vào fixed-capacity locked secret buffer.
4. Decrypt từng declared protected file/record theo deterministic order.
5. Chỉ ghi record đã authenticated vào `.partial`.
6. Verify container hash, plaintext full-file hash/size, record count và exact
   EOF; atomic publish output.
7. Verify model index chỉ tham chiếu declared output trong model root.
8. Zeroize DEK, crypto state và owned staging buffers trước vLLM load.

Không truyền raw DEK qua Python. `zeroize` phải bao phủ AES expanded state,
transport/FFI copies và allocation capacity do core sở hữu. Core hiện đặt
production `SecretDek` trong một private anonymous page, yêu cầu `mlock` và
`MADV_DONTDUMP`, rồi zeroize, `munlock` và `munmap` bằng RAII. Nếu kernel hoặc
`RLIMIT_MEMLOCK` từ chối, key release trả `SECRET_MEMORY_UNAVAILABLE` và không
fallback sang pageable memory. Cơ chế này không bao phủ mọi copy trong
TPM/crypto library hoặc engine. Không claim xóa registers, kernel/TPM internal
state, SIGKILL copies hoặc GPU memory.

### 8.6 Engine load, registration và serving

Session owner giữ handle trong outer `try/finally` bao quanh toàn worker:

1. `create_engine_config()` dùng verified paths.
2. Revalidate model/tokenizer/weights paths sau vLLM normalization.
3. Tạo AsyncLLM và đợi engine/TP workers ready.
4. Đăng ký Dynamo model bằng UUID session directory; public served identity vẫn là
   original/explicit served name.
5. Model-card publication chỉ thấy declared model metadata; session state,
   license và wrapped key không nằm trong directory đó.
6. Chỉ công bố readiness sau engine, workers và `register_model()` thành công.

Offline perpetual V1 không có renewal state. V1 giữ plaintext tmpfs đến worker
shutdown để bao phủ possible reread, sleep/wake và loader behavior. Điều này kéo
dài exposure với process cùng UID; early unlink chỉ được bật sau evidence cho
từng vLLM version/load mode.

## 9. Session state và cleanup

```text
BOOTSTRAP
  ├─ PLAIN → EXISTING_FLOW
  └─ SECURE_CANDIDATE
       → PACKAGE_VERIFIED
       → LICENSE_VERIFIED
       → VOLUME_OWNED
       → METADATA_STAGED
       → EFFECTIVE_CONFIG_APPROVED
       → KEY_RELEASED
       → MODEL_MATERIALIZED
       → ENGINE_LOADING
       → DYNAMO_REGISTERED
       → SERVING
       → DRAINING
       → CLEANUP
```

Secret guard bắt đầu từ allocation/transport trước `KEY_RELEASED`, không chờ
session state. Mọi transition có cancellation path.

### 9.1 Ownership và stale recovery

- Acquire exclusive `flock` trên `.owner.lock` trước sweep/session creation.
- Một volume chỉ có một live owner trong V1.
- Startup sweep chỉ đi dưới verified tmpfs root bằng safe FD operations.
- Sweep chạy nếu secure mount hiện diện và có stale protected-session state,
  kể cả lần khởi động sau dùng plain model; plain model không fail nếu mount
  không tồn tại.
- Không dùng PID hoặc mtime làm bằng chứng duy nhất khi nhiều owner được support
  trong tương lai.
- Entry lạ, symlink hoặc live-owner conflict làm fail recovery; không recursive
  follow.

### 9.2 Orderly shutdown/error

1. Dừng admission; bounded drain nếu serving.
2. Cancel engine operations.
3. Dừng và reap toàn bộ loader-owned vLLM child process.
4. Đóng mmap/file handles do loader/session quản lý.
5. Zeroize owned secret/cipher buffers còn tồn tại.
6. Xóa model files, state và session directory bằng safe FD traversal.
7. Verify session absent; release owner lock.

Cleanup failure giữ worker unready và yêu cầu replace/delete Pod; không retry
load trong volume không xác định. `SIGTERM`/`SIGINT` chỉ signal orderly shutdown;
không chạy complex filesystem cleanup trong async signal handler.

`SIGKILL` không cho process cleanup. Container restart trong cùng Pod vẫn giữ
`emptyDir`, nên startup recovery bắt buộc. Pod deletion request chưa chứng minh
RAM được sanitize ngay; operator phải verify process termination và volume
unmount, đặc biệt khi node unreachable.

Unlink/Pod deletion/zeroization không được mô tả là physical RAM sanitization
tuyệt đối.

## 10. Error và audit contract

Stable external error classes:

- `SECURE_PACKAGE_INVALID`
- `MANIFEST_SIGNATURE_INVALID`
- `PACKAGE_INTEGRITY_INVALID`
- `LICENSE_INVALID`
- `LICENSE_SIGNATURE_INVALID`
- `LICENSE_BINDING_MISMATCH`
- `DECRYPTION_FAILED`
- `TMPFS_INVALID`
- `TMPFS_INSUFFICIENT`
- `SESSION_CONFLICT`
- `TPM_UNAVAILABLE`
- `TPM_AUTHORIZATION_FAILED`
- `SECRET_MEMORY_UNAVAILABLE`
- `RUNTIME_UNSUPPORTED`
- `MODEL_PROTECTION_CONFIG_INVALID`
- `MODEL_PROTECTION_MODE_UNSUPPORTED`
- `CLEANUP_FAILED`
- `SECURE_IO_ERROR`

Wrong DEK, corrupted ciphertext và tag failure dùng cùng external decryption
class; không tạo oracle chi tiết. `ProtectionError::code()` là source của stable
code và `Display` không render OS error/path. Internal structured logs chỉ lấy
timestamp, correlation ID, artifact/package opaque ID, key ID, state, stable
error code và reason tĩnh từ `sanitized_reason()`; không log `Debug` hoặc raw
source error tại protection boundary.

Không log:

- DEK/KEK/DUK secret;
- wrapped key blob hoặc TPM authorization payload đầy đủ;
- plaintext bytes/path contents;
- raw attestation chứa sensitive inventory;
- nonce/tag collection;
- license/customer identifiers không cần thiết.

Audit events: activation, license verify result, key-release result, package
digest, session state transition, load readiness, shutdown và cleanup result.
Audit sink không nằm trong secure model directory.

## 11. Rollback, rotation và recovery

- Package/license/container trust roots độc lập và versioned.
- New signer metadata phải được current trusted root authorize, có version và
  expiry chống rollback.
- License/package chứa `minimum_runtime_version`; deployment admission hoặc key
  release policy enforce floor. Old runtime không có detector không thể tự bảo
  đảm fail closed.
- Issuer backup giữ wrapped DEK records, encrypted key directory và key metadata
  trong hai backup domain riêng; restore drill không xuất plaintext DEK ra file.
- Compromised package signer: revoke key ID, stop issue license, rebuild artifact
  với fresh DEK/signing key nếu cần.
- Compromised license signer: rotate/revoke license key, tăng generation và
  reissue; offline perpetual copies cũ không thể bị thu hồi ngay.
- Lost issuer DEK record: artifact không thể reissue sang máy mới; khôi phục
  backup hoặc rebuild/re-encrypt artifact.
- Rollback image chỉ được chọn trong compatibility/minimum-version policy.

Không cần đưa toàn bộ TUF framework vào V1; áp dụng các nguyên tắc signed trusted
metadata, version, threshold/rotation khi risk assessment yêu cầu.

## 12. Source boundaries đề xuất

Không tạo nhiều manager/interface cho một implementation. Vertical slice cần:

```text
lib/model-protection/                 # Rust format, signatures, safe I/O, TPM/decrypt
lib/bindings/python/                  # opaque bootstrap/session binding
components/src/dynamo/vllm/
├── protection_bootstrap.py           # two-phase orchestration
└── main.py                           # session lifetime + verified paths
lib/model-protection/src/bin/         # software-key packager/license-issuer CLI
```

Điểm tích hợp cần sửa/kiểm tra:

- `components/src/dynamo/vllm/main.py`: bootstrap trước `parse_args`, fetch,
  snapshot; outer session guard; resolved registration path.
- `components/src/dynamo/vllm/backend_args.py`: runtime-config option;
  `protection_bootstrap.py`: effective config allowlist và served identity.
- `components/src/dynamo/vllm/worker_factory.py`: protected worker không đăng
  ký profile/sleep/wake/cache/LoRA/weight-update routes; chỉ giữ liveness và
  model-taint route.
- `components/src/dynamo/common/utils/namespace.py` và runtime args: một
  effective namespace contract.
- Registration nhận UUID session directory chỉ chứa declared metadata và
  weights; license/session control không được copy vào directory này.
- Kubernetes operator: dedicated memory volume, concrete path, UID ownership,
  memory/security policy và node TPM scheduling/access.

Đây là boundary đã được implement trong nhánh; ownership cuối cùng
vẫn theo DEP/CODEOWNERS review.

## 13. Kế hoạch triển khai đã cập nhật

### Phase 0 — Freeze security contracts

- Viết DEP và threat model.
- Freeze package/license schema, canonical signed bytes và limits.
- Freeze record encoding, fresh-DEK/counter/nonce/AAD contract.
- Freeze TPM enrollment, AK/DUK certification, policy authorization và RMA.
- Freeze strict host persistence profile và supported vLLM matrix.
- Tạo golden vectors độc lập giữa packager và loader.

Exit gate: security review phê duyệt contract; chưa cần GPU.

### Phase 1 — Software issuer và offline TPM vertical slice

- Pack tiny safetensors artifact với fresh DEK.
- Implement encrypted-file key loading cho Ed25519, ECDSA P-256 và AES-256 KEK;
  reject symlink, sai owner/mode, sai format/size và key ID. Runbook/audit
  xác nhận key directory nằm trên encrypted offline volume; code không cố suy
  ra encryption-at-rest từ pathname.
- Lưu AES-KWP-wrapped DEK record; issue offline perpetual TPM-bound license.
- Implement Rust strict parser, signature, safe I/O, TPM unwrap, AEAD records,
  secret guards và cleanup.
- Implement two-phase Dynamo bootstrap cho một local vLLM configuration.
- Load, inference và shutdown cleanup trên một GPU/TPM node.

Exit gate: package→issue→TPM unwrap→decrypt happy path và negative
key/package/license/path tests có evidence. PKCS#11/HSM không phải gate.

### Phase 2 — Dynamo/vLLM compatibility

- Plain model regression không đổi behavior.
- Negative test chứng minh TP/PP/DP khác 1 bị reject trước key release.
- Engine/config path rewrite và post-normalization validation.
- Registration/public metadata tests.
- Verify mọi rejected mode/entrypoint fail trước key release.
- Measure tmpfs/RSS/load peak và lifecycle rereads.

Exit gate: versioned compatibility matrix cho vLLM `0.30.0` và supported flags.

### Phase 3 — Failure and deployment hardening

- Failure injection ở mọi state: cancellation, ENOSPC/ENOMEM, wrong tag, engine
  child crash, registration failure, SIGTERM/SIGKILL và container restart.
- Kubernetes dedicated tmpfs, UID/volume ownership, memory requests/limits,
  no-swap/dump policy và TPM device scheduling.
- Image/package/license signing, encrypted key-directory rotation,
  backup/restore, reactivation và
  rollback-floor drills.
- Stable errors, sanitized logs và operator feedback.

Exit gate: recovery evidence và security review sign-off.

### Phase 4 — Online entitlement nếu sản phẩm yêu cầu

- RATS-style fresh attestation và recipient-bound channel.
- Serving lease/renewal/deadline/grace/admission/drain semantics.
- Revocation and outage drills.
- Không file-key fallback và không reuse offline semantics dưới tên online.

### Phase 5 — Chỉ sau evidence

- Early tmpfs unlink sau model load.
- Disaggregated multi-Pod/node authorization.
- Snapshot/CRIU.
- Protected LoRA/runtime weight update.
- Direct authenticated tensor/GPU loading.

## 14. Acceptance matrix

| Nhóm | Tests/evidence bắt buộc |
|---|---|
| Plain compatibility | HF ID/local safetensors hoạt động; không TPM/license/tmpfs allocation |
| Detector/parser | Plain `.bin`; mixed/missing markers; duplicate JSON key; bounds/fuzz corpus |
| Package crypto | Golden vectors; wrong signature/tag/hash; reorder/append/truncate; counter limits |
| TPM activation | Replay/substituted DUK/untrusted AK/wrong package; bypass authorization path |
| Filesystem | Parent symlink, magic link, nested mount, source mutation, collision, hard link |
| Memory | Host/cgroup pressure, swap/dump verification, actual peak theo phase |
| Dynamo/vLLM | Path/served-name/registration correctness; all allowlist/reject combinations |
| Metadata | Model card/cache chỉ nhận declared public files; multimodal processor vẫn hoạt động |
| Lifecycle | Failure trước/sau unwrap, decrypt, engine init, registration; child reaping |
| Recovery | Protected container restart, stale session, owner conflict, protected-to-plain Pod replacement |
| Operations | License reissue, TPM clear/RMA, key rotation/revocation, encrypted key-store restore, rollback floor |

Security path cần property/fuzz tests cho parser và path normalization, golden
crypto vectors, fault injection và real-node integration. Unit test không được
dùng để claim perfect memory zeroization, root resistance hoặc physical RAM
sanitization.

## 15. Trạng thái quyết định

Đã chốt trong V2:

- V1 offline TPM, license perpetual và controlled reactivation;
- one fresh DEK/customer/model-version artifact;
- separate package/license/TPM-policy/container trust keys;
- one file entry with ordered independent AEAD records;
- strict safe-path and no-sensitive-persistence profile;
- two-phase bootstrap và explicit secure backend allowlist;
- single-GPU vLLM safetensors with `TP=PP=DP=1`;
- plaintext tmpfs giữ đến shutdown cho tới khi có reread evidence;
- public metadata allowlist và session state tách khỏi model directory.
- record header/AAD/domain encoding và exact-byte manifest signature;
- secure namespace grammar 1-128 ASCII bytes như mục 6.2;
- parser/record/safe-I/O, TPM, PyO3, packager/issuer và deployment reference
  trong working tree.
- HSM/PKCS#11 bị loại khỏi active V1; software issuer key provider đã được
  implement trong hai CLI. Key-directory operations và target acceptance vẫn là
  product gates.

Còn cần chứng minh trước production release:

1. Physical TPM acceptance cho public-area/Name, cpHash/approved-policy,
   `PolicyAuthorize` và reset/reprovision; đồng thời security acceptance cho
   offline issuer host, encrypted key directory, backup và rotation.
2. Model-specific public/protected metadata inventory.
3. RAM margin và size limits từ benchmark trên target hardware.
4. Target-container acceptance cho `mlock`/`MADV_DONTDUMP`, nonzero
   `RLIMIT_MEMLOCK`, swap/core-dump/hibernation policy và các plaintext buffer
   ngoài dedicated DEK page.
5. Whether/when online calendar-expiry profile is a product requirement.
6. Plain/secure local-or-Docker end-to-end inference và cross-server denial.
   Kubernetes mount/cgroup/Pod crash evidence chỉ bắt buộc cho Kubernetes profile.

## 16. Tài liệu tham chiếu

- [DEP #14764: Protected model loading for on-premises deployments](https://github.com/ai-dynamo/dynamo/issues/14764)
- [Security review and findings](model-protection-security-review.md)
- [NIST Key Management Guidelines](https://csrc.nist.gov/projects/key-management/key-management-guidelines)
- [NIST SP 800-38D: GCM and GMAC](https://csrc.nist.gov/pubs/sp/800/38/d/final)
- [RFC 8032: Ed25519](https://www.rfc-editor.org/info/rfc8032/)
- [RFC 9334: RATS Architecture](https://www.rfc-editor.org/rfc/rfc9334.html)
- [TPM PolicyAuthorize](https://tpm2-tools.readthedocs.io/en/latest/man/tpm2_policyauthorize.1/)
- [TPM object creation and policy-only authorization](https://tpm2-tools.readthedocs.io/en/latest/man/tpm2_create.1/)
- [TPM RSA decrypt/OAEP label semantics](https://tpm2-tools.readthedocs.io/en/stable/man/tpm2_rsadecrypt.1/)
- [TCG TPM 2.0 provisioning guidance](https://trustedcomputinggroup.org/resource/tcg-tpm-v2-0-provisioning-guidance/)
- [TPM quote](https://tpm2-tools.readthedocs.io/en/latest/man/tpm2_quote.1/)
- [TPM certify](https://tpm2-tools.readthedocs.io/en/latest/man/tpm2_certify.1/)
- [TPM clock semantics](https://tpm2-tools.readthedocs.io/en/latest/man/tpm2_readclock.1/)
- [Linux tmpfs](https://docs.kernel.org/filesystems/tmpfs.html)
- [Linux openat2](https://man7.org/linux/man-pages/man2/openat2.2.html)
- [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [Kubernetes emptyDir](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir)
- [Rust zeroize limits](https://docs.rs/zeroize/latest/zeroize/)
- [vLLM 0.30.0 EngineArgs](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py)
- [TUF specification](https://theupdateframework.github.io/specification/latest/)
