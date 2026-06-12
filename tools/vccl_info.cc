/* SPDX-License-Identifier: Apache-2.0
 *
 * vccl-info — RDMA device/port/GID survey for cluster bring-up.
 *
 * Prints every verbs device with port state, link layer, MTU, and the GID
 * table annotated with types and which entry vccl would auto-select.
 * Exit code 0 when at least one active port exists.
 */
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstring>

namespace {

bool gidIsZero(const ibv_gid& g) {
  for (unsigned char b : g.raw)
    if (b != 0) return false;
  return true;
}

bool gidIsIpv4Mapped(const ibv_gid& g) {
  for (int i = 0; i < 10; i++)
    if (g.raw[i] != 0) return false;
  return g.raw[10] == 0xff && g.raw[11] == 0xff;
}

void gidToStr(const ibv_gid& g, char* out, size_t len) {
  if (gidIsIpv4Mapped(g)) {
    char v4[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, g.raw + 12, v4, sizeof(v4));
    snprintf(out, len, "::ffff:%s", v4);
  } else {
    inet_ntop(AF_INET6, g.raw, out, len);
  }
}

const char* gidType(const char* dev, uint8_t port, int idx) {
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d", dev, port,
           idx);
  static char buf[32];
  FILE* f = fopen(path, "r");
  if (f == nullptr) return "?";
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = 0;
  if (char* nl = strchr(buf, '\n')) *nl = 0;
  return buf;
}

const char* portState(int state) {
  switch (state) {
    case IBV_PORT_DOWN: return "DOWN";
    case IBV_PORT_INIT: return "INIT";
    case IBV_PORT_ARMED: return "ARMED";
    case IBV_PORT_ACTIVE: return "ACTIVE";
    default: return "?";
  }
}

}  // namespace

int main() {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (list == nullptr || num == 0) {
    printf("no RDMA devices found\n");
    printf("hints: modprobe ib_uverbs; modprobe <nic driver> (mlx5_ib...)\n");
    printf("       soft-RoCE: modprobe rdma_rxe && "
           "rdma link add rxe0 type rxe netdev <if>\n");
    return 1;
  }

  int activePorts = 0;
  for (int i = 0; i < num; i++) {
    const char* name = ibv_get_device_name(list[i]);
    ibv_context* ctx = ibv_open_device(list[i]);
    if (ctx == nullptr) {
      printf("%s: open failed\n", name);
      continue;
    }
    ibv_device_attr dattr{};
    ibv_query_device(ctx, &dattr);
    printf("%s (guid %016llx)\n", name,
           static_cast<unsigned long long>(dattr.node_guid));
    for (uint8_t p = 1; p <= dattr.phys_port_cnt; p++) {
      ibv_port_attr pattr{};
      if (ibv_query_port(ctx, p, &pattr) != 0) continue;
      bool roce = pattr.link_layer == IBV_LINK_LAYER_ETHERNET;
      printf("  port %u: %s, %s, active_mtu %d\n", p,
             portState(pattr.state), roce ? "RoCE/Ethernet" : "InfiniBand",
             128 << pattr.active_mtu);
      if (pattr.state == IBV_PORT_ACTIVE) activePorts++;
      for (int g = 0; g < pattr.gid_tbl_len; g++) {
        ibv_gid gid{};
        if (ibv_query_gid(ctx, p, g, &gid) != 0 || gidIsZero(gid)) continue;
        char str[64];
        gidToStr(gid, str, sizeof(str));
        const char* type = gidType(name, p, g);
        bool preferred = roce && gidIsIpv4Mapped(gid) &&
                         strstr(type, "RoCE v2") != nullptr;
        printf("    gid %2d: %-40s [%s]%s\n", g, str, type,
               preferred ? "  <- vccl default" : "");
      }
    }
    ibv_close_device(ctx);
  }
  ibv_free_device_list(list);
  if (activePorts == 0) printf("no ACTIVE ports\n");
  return activePorts > 0 ? 0 : 1;
}
