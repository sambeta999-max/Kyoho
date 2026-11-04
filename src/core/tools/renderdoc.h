// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Tools {

class RenderdocAPI {
public:
    explicit RenderdocAPI();
    ~RenderdocAPI();

    void ToggleCapture();

    /// Captures the next `frames` presented frames (RenderDoc's TriggerMultiFrameCapture)
    void TriggerCapture(unsigned frames);

private:
    void* rdoc_api{};
    bool is_capturing{false};
};

} // namespace Tools
