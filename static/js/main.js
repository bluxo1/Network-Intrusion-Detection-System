/* ============================================================
   NIDS-PyTorch  |  form presets + helpers
   Each preset is a REAL NSL-KDD record (all 41 features) that the
   trained model classifies as the labelled class, so one click loads a
   representative connection instead of typing 41 fields by hand.
   ============================================================ */

const PRESETS = {
  // Benign FTP-data transfer.
  normal: {
    duration: 0, protocol_type: "tcp", service: "ftp_data", flag: "SF",
    src_bytes: 491, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0, hot: 0,
    num_failed_logins: 0, logged_in: 0, num_compromised: 0, root_shell: 0,
    su_attempted: 0, num_root: 0, num_file_creations: 0, num_shells: 0,
    num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
    count: 2, srv_count: 2, serror_rate: 0, srv_serror_rate: 0, rerror_rate: 0,
    srv_rerror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, srv_diff_host_rate: 0,
    dst_host_count: 150, dst_host_srv_count: 25, dst_host_same_srv_rate: 0.17,
    dst_host_diff_srv_rate: 0.03, dst_host_same_src_port_rate: 0.17,
    dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 0,
    dst_host_srv_serror_rate: 0, dst_host_rerror_rate: 0.05, dst_host_srv_rerror_rate: 0,
  },
  // Neptune SYN flood: half-open connections, SYN-error rates at 1.0.
  dos: {
    duration: 0, protocol_type: "tcp", service: "private", flag: "S0",
    src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0, hot: 0,
    num_failed_logins: 0, logged_in: 0, num_compromised: 0, root_shell: 0,
    su_attempted: 0, num_root: 0, num_file_creations: 0, num_shells: 0,
    num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
    count: 123, srv_count: 6, serror_rate: 1, srv_serror_rate: 1, rerror_rate: 0,
    srv_rerror_rate: 0, same_srv_rate: 0.05, diff_srv_rate: 0.07, srv_diff_host_rate: 0,
    dst_host_count: 255, dst_host_srv_count: 26, dst_host_same_srv_rate: 0.1,
    dst_host_diff_srv_rate: 0.05, dst_host_same_src_port_rate: 0,
    dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 1,
    dst_host_srv_serror_rate: 1, dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0,
  },
  // ICMP echo sweep (ipsweep-style reconnaissance).
  probe: {
    duration: 0, protocol_type: "icmp", service: "eco_i", flag: "SF",
    src_bytes: 18, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0, hot: 0,
    num_failed_logins: 0, logged_in: 0, num_compromised: 0, root_shell: 0,
    su_attempted: 0, num_root: 0, num_file_creations: 0, num_shells: 0,
    num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
    count: 1, srv_count: 1, serror_rate: 0, srv_serror_rate: 0, rerror_rate: 0,
    srv_rerror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, srv_diff_host_rate: 0,
    dst_host_count: 1, dst_host_srv_count: 16, dst_host_same_srv_rate: 1,
    dst_host_diff_srv_rate: 0, dst_host_same_src_port_rate: 1,
    dst_host_srv_diff_host_rate: 1, dst_host_serror_rate: 0,
    dst_host_srv_serror_rate: 0, dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0,
  },
  // R2L: authenticated ftp-data session flagged as warez-style unauthorized access.
  r2l: {
    duration: 0, protocol_type: "tcp", service: "ftp_data", flag: "SF",
    src_bytes: 334, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0, hot: 0,
    num_failed_logins: 0, logged_in: 1, num_compromised: 0, root_shell: 0,
    su_attempted: 0, num_root: 0, num_file_creations: 0, num_shells: 0,
    num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
    count: 2, srv_count: 2, serror_rate: 0, srv_serror_rate: 0, rerror_rate: 0,
    srv_rerror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, srv_diff_host_rate: 0,
    dst_host_count: 2, dst_host_srv_count: 20, dst_host_same_srv_rate: 1,
    dst_host_diff_srv_rate: 0, dst_host_same_src_port_rate: 1,
    dst_host_srv_diff_host_rate: 0.2, dst_host_serror_rate: 0,
    dst_host_srv_serror_rate: 0, dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0,
  },
  // U2R: telnet session with a root shell obtained and many root accesses.
  u2r: {
    duration: 98, protocol_type: "tcp", service: "telnet", flag: "SF",
    src_bytes: 621, dst_bytes: 8356, land: 0, wrong_fragment: 0, urgent: 1, hot: 1,
    num_failed_logins: 0, logged_in: 1, num_compromised: 5, root_shell: 1,
    su_attempted: 0, num_root: 14, num_file_creations: 1, num_shells: 0,
    num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
    count: 1, srv_count: 1, serror_rate: 0, srv_serror_rate: 0, rerror_rate: 0,
    srv_rerror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, srv_diff_host_rate: 0,
    dst_host_count: 255, dst_host_srv_count: 4, dst_host_same_srv_rate: 0.02,
    dst_host_diff_srv_rate: 0.02, dst_host_same_src_port_rate: 0,
    dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 0,
    dst_host_srv_serror_rate: 0, dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0,
  },
};

// Snapshot of the server-rendered defaults so "Reset" can restore them.
const INITIAL = {};
document.querySelectorAll("#nids-form [name]").forEach((el) => {
  INITIAL[el.name] = el.value;
});

function setField(name, value) {
  const el = document.querySelector(`#nids-form [name="${name}"]`);
  if (el) el.value = value;
}

function applyPreset(preset) {
  // Reset to defaults first, then overlay the preset's fields.
  Object.entries(INITIAL).forEach(([k, v]) => setField(k, v));
  Object.entries(preset).forEach(([k, v]) => setField(k, v));
}

document.querySelectorAll(".preset[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = PRESETS[btn.dataset.preset];
    if (preset) applyPreset(preset);
  });
});

const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    Object.entries(INITIAL).forEach(([k, v]) => setField(k, v));
  });
}
