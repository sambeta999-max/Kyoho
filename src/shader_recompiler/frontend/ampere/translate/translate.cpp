// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/decode.h"
#include "shader_recompiler/frontend/ampere/location.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ampere/translate/translate.h"

namespace Shader::Ampere {
namespace {

void Dispatch(TranslatorVisitor& v, Opcode opcode, Instruction insn) {
    switch (opcode) {
    case Opcode::IABS:
        return v.IABS(insn);
    case Opcode::IADD3:
        return v.IADD3(insn);
    case Opcode::UIADD3:
        return v.UIADD3(insn);
    case Opcode::IMAD:
        return v.IMAD(insn);
    case Opcode::UIMAD:
        return v.UIMAD(insn);
    case Opcode::IMAD_WIDE:
        return v.IMAD_WIDE(insn);
    case Opcode::UIMAD_WIDE:
        return v.UIMAD_WIDE(insn);
    case Opcode::IMAD_HI:
        return v.IMAD_HI(insn);
    case Opcode::IMNMX:
        return v.IMNMX(insn);
    case Opcode::LDGSTS:
        return v.LDGSTS(insn);
    case Opcode::IDP:
        return v.IDP(insn);
    case Opcode::ISETP:
        return v.ISETP(insn);
    case Opcode::UISETP:
        return v.UISETP(insn);
    case Opcode::LOP3:
        return v.LOP3(insn);
    case Opcode::ULOP3:
        return v.ULOP3(insn);
    case Opcode::PLOP3:
        return v.PLOP3(insn);
    case Opcode::UPLOP3:
        return v.UPLOP3(insn);
    case Opcode::R2P:
        return v.R2P(insn);
    case Opcode::UR2UP:
        return v.UR2UP(insn);
    case Opcode::P2R:
        return v.P2R(insn);
    case Opcode::UP2UR:
        return v.UP2UR(insn);
    case Opcode::SGXT:
        return v.SGXT(insn);
    case Opcode::USGXT:
        return v.USGXT(insn);
    case Opcode::SHF:
        return v.SHF(insn);
    case Opcode::USHF:
        return v.USHF(insn);
    case Opcode::SEL:
        return v.SEL(insn);
    case Opcode::USEL:
        return v.USEL(insn);
    case Opcode::POPC:
        return v.POPC(insn);
    case Opcode::UPOPC:
        return v.UPOPC(insn);
    case Opcode::FLO:
        return v.FLO(insn);
    case Opcode::UFLO:
        return v.UFLO(insn);
    case Opcode::BREV:
        return v.BREV(insn);
    case Opcode::UBREV:
        return v.UBREV(insn);
    case Opcode::PRMT:
        return v.PRMT(insn);
    case Opcode::UPRMT:
        return v.UPRMT(insn);
    case Opcode::LEA:
        return v.LEA(insn);
    case Opcode::ULEA:
        return v.ULEA(insn);
    case Opcode::F2F:
        return v.F2F(insn);
    case Opcode::FRND:
        return v.FRND(insn);
    case Opcode::F2I:
        return v.F2I(insn);
    case Opcode::F2I_F64:
        return v.F2I_F64(insn);
    case Opcode::I2F_64:
        return v.I2F_64(insn);
    case Opcode::BPT:
        return v.BPT(insn);
    case Opcode::I2F:
        return v.I2F(insn);
    case Opcode::I2FP:
        return v.I2FP(insn);
    case Opcode::LDG:
        return v.LDG(insn);
    case Opcode::STG:
        return v.STG(insn);
    case Opcode::LDS:
        return v.LDS(insn);
    case Opcode::STS:
        return v.STS(insn);
    case Opcode::LDL:
        return v.LDL(insn);
    case Opcode::STL:
        return v.STL(insn);
    case Opcode::ALD:
        return v.ALD(insn);
    case Opcode::ISBERD:
        return v.ISBERD(insn);
    case Opcode::HMMA:
        return v.HMMA(insn);
    case Opcode::LDSM:
        return v.LDSM(insn);
    case Opcode::MOVM:
        return v.MOVM(insn);
    case Opcode::OUT:
        return v.OUT(insn);
    case Opcode::LEPC:
        return v.LEPC(insn);
    case Opcode::BMOV_64:
        // Only writes ATEXIT_PC, the handler EXIT jumps to. Geometry programs point it at
        // OUT.FINAL; ending the invocation already ends the primitive on the host.
        return;
    case Opcode::AST:
        return v.AST(insn);
    case Opcode::IPA:
        return v.IPA(insn);
    case Opcode::TEX:
    case Opcode::TEX_BINDLESS:
        return v.TEX(insn);
    case Opcode::TLD:
    case Opcode::TLD_BINDLESS:
        return v.TLD(insn);
    case Opcode::TLD4:
    case Opcode::TLD4_BINDLESS:
        return v.TLD4(insn);
    case Opcode::TMML:
    case Opcode::TMML_BINDLESS:
        return v.TMML(insn);
    case Opcode::TXD:
    case Opcode::TXD_BINDLESS:
        return v.TXD(insn);
    case Opcode::TXQ:
    case Opcode::TXQ_BINDLESS:
        return v.TXQ(insn);
    case Opcode::SHFL:
        return v.SHFL(insn);
    case Opcode::VOTE:
        return v.VOTE(insn);
    case Opcode::VOTEU:
        return v.VOTEU(insn);
    case Opcode::REDUX:
        return v.REDUX(insn);
    case Opcode::MATCH:
        return v.MATCH(insn);
    case Opcode::ATOMG:
        return v.ATOMG(insn);
    case Opcode::RED:
        return v.RED(insn);
    case Opcode::ATOMS:
        return v.ATOMS(insn);
    case Opcode::SULD_P:
    case Opcode::SULD_P_BINDLESS:
    case Opcode::SULD_D:
    case Opcode::SULD_D_BINDLESS:
        return v.SULD(insn);
    case Opcode::SUST_P:
    case Opcode::SUST_P_BINDLESS:
    case Opcode::SUST_D:
    case Opcode::SUST_D_BINDLESS:
        return v.SUST(insn);
    case Opcode::SUATOM:
        return v.SUATOM(insn);
    case Opcode::SURED:
        return v.SURED(insn);
    case Opcode::FADD:
        return v.FADD(insn);
    case Opcode::FSWZADD:
        return v.FSWZADD(insn);
    case Opcode::FMUL:
        return v.FMUL(insn);
    case Opcode::FFMA:
        return v.FFMA(insn);
    case Opcode::FMNMX:
        return v.FMNMX(insn);
    case Opcode::FSET:
        return v.FSET(insn);
    case Opcode::FSETP:
        return v.FSETP(insn);
    case Opcode::FSEL:
        return v.FSEL(insn);
    case Opcode::MUFU:
        return v.MUFU(insn);
    case Opcode::HADD2:
        return v.HADD2(insn);
    case Opcode::HMUL2:
        return v.HMUL2(insn);
    case Opcode::HFMA2:
        return v.HFMA2(insn);
    case Opcode::HMNMX2:
        return v.HMNMX2(insn);
    case Opcode::HSET2:
        return v.HSET2(insn);
    case Opcode::HSETP2:
        return v.HSETP2(insn);
    case Opcode::F2FP:
        return v.F2FP(insn);
    case Opcode::MOV:
        return v.MOV(insn);
    case Opcode::UMOV:
        return v.UMOV(insn);
    case Opcode::R2UR:
        return v.R2UR(insn);
    case Opcode::S2R:
        return v.S2R(insn);
    case Opcode::S2UR:
        return v.S2UR(insn);
    case Opcode::CS2R:
        return v.CS2R(insn);
    case Opcode::LDC:
        return v.LDC(insn);
    case Opcode::ULDC:
    case Opcode::ULDC_INDEXED:
        return v.ULDC(insn);
    case Opcode::EXIT:
        return v.EXIT(insn);
    case Opcode::BAR:
        return v.BAR(insn);
    case Opcode::CSMTEST:
        return v.CSMTEST(insn);
    case Opcode::MEMBAR:
        return v.MEMBAR(insn);
    case Opcode::BRA:
    case Opcode::BRX:
    case Opcode::BRXU:
    case Opcode::CALL_REL:
    case Opcode::RET:
    case Opcode::KILL:
        // Handled by the control flow graph
        return;
    case Opcode::NOP:
    case Opcode::BSSY:
    case Opcode::BSYNC:
    case Opcode::BREAK:
    case Opcode::WARPSYNC:
    case Opcode::YIELD:
    case Opcode::DEPBAR:
    case Opcode::LDGDEPBAR:
    case Opcode::ARRIVES_LDGSTSBAR:
    case Opcode::ERRBAR:
    case Opcode::CCTL:
        // Scheduling, convergence and cache hints have no effect on the host
        return;
    default:
        throw NotImplementedException("Instruction {}", opcode);
    }
}

} // Anonymous namespace

void Translate(Environment& env, IR::Block* block, Maxwell::Location begin,
               Maxwell::Location end) {
    TranslatorVisitor visitor{env, *block};
    for (Maxwell::Location pc = begin; pc != end; ++pc) {
        const u32 offset{ToOffset(pc)};
        const Instruction insn{env.ReadInstruction(offset), env.ReadInstruction(offset + 8)};
        if (insn.Pred() == 7 && insn.PredNegated()) {
            // Guarded by !PT, never executed
            continue;
        }
        const Opcode opcode{Decode(insn)};
        if (opcode == Opcode::ISBEWR) {
            throw NotImplementedException("Task/Mesh shader instruction {}", opcode);
        }
        if (opcode == Opcode::IMMA || opcode == Opcode::DMMA || opcode == Opcode::BMMA) {
            throw NotImplementedException("Tensor matrix multiply instruction {}", opcode);
        }
        try {
            Dispatch(visitor, opcode, insn);
        } catch (const NotImplementedException& exception) {
            // Keep translating the rest of the program; the instruction's results are not written
            LOG_ERROR(Shader, "SM86 {:#06x}: {}", offset, exception.what());
            if (unimplemented_observer) {
                unimplemented_observer(exception.what());
            }
        }
    }
}

} // namespace Shader::Ampere
