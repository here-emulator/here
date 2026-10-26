use core::panic;
use regex::Regex;
use serde_json::Value;
use std::{collections::HashMap, env, fs, path::PathBuf};

fn to_bits(s: &str) -> u64 {
    u64::from_str_radix(s, 2).unwrap_or(0)
}

fn get_instr_bits(s: &str, low: usize, high: usize) -> &str {
    let idx1 = s.len() - high - 1;
    let idx2 = s.len() - low - 1;
    &s[idx1..=idx2]
}

fn get_opcode(s: &str) -> u64 {
    to_bits(get_instr_bits(s, 0, 6))
}

fn get_funct3(s: &str) -> u64 {
    to_bits(get_instr_bits(s, 12, 14))
}

fn hex_to_u64(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
}

fn instr_object<'a>(value: &'a Value, path: &PathBuf) -> &'a serde_json::Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{} must contain a JSON object", path.display()))
}

fn instr_array<'a>(instr: &'a Value, instr_name: &str, field: &str) -> &'a Vec<Value> {
    instr[field].as_array().unwrap_or_else(|| {
        panic!(
            "Instruction `{}` field `{}` must be an array",
            instr_name, field
        )
    })
}

fn instr_str<'a>(instr: &'a Value, instr_name: &str, field: &str) -> &'a str {
    instr[field].as_str().unwrap_or_else(|| {
        panic!(
            "Instruction `{}` field `{}` must be a string",
            instr_name, field
        )
    })
}

fn value_str<'a>(value: &'a Value, instr_name: &str, field: &str) -> &'a str {
    value.as_str().unwrap_or_else(|| {
        panic!(
            "Instruction `{}` field `{}` must contain strings",
            instr_name, field
        )
    })
}

fn is_vector_load(encoding: &str) -> bool {
    let func3 = get_funct3(encoding);
    let func3_is_vector = match func3 {
        0b000 | 0b101 | 0b110 | 0b111 => true,
        _ => false,
    };
    get_opcode(encoding) == 0b0000111 && func3_is_vector
}

fn is_vector_store(encoding: &str) -> bool {
    let func3 = get_funct3(encoding);
    let func3_is_vector = match func3 {
        0b000 | 0b101 | 0b110 | 0b111 => true,
        _ => false,
    };
    get_opcode(encoding) == 0b0100111 && func3_is_vector
}

fn is_vector_arith(encoding: &str) -> bool {
    get_opcode(encoding) == 0b1010111 // vector arithmetic instructions
}

fn vector_ignore_instr(name: &str, encoding: &str) -> bool {
    let re_float_ignore = Regex::new("v.*red.*").unwrap();
    let is_float_func3 = match get_funct3(encoding) {
        0b001 | 0b101 => true,
        _ => false,
    }; // floating-point instructions
    let is_float = (is_vector_arith(encoding) && is_float_func3) || re_float_ignore.is_match(name);

    is_float
}

fn get_instr_type(fields: Vec<&str>, name: &str, _ext: &str, encoding: &str) -> &'static str {
    if fields == ["rd", "rs1", "rs2"] || fields == ["rd", "rs1"] || fields == ["rs1", "rs2"] {
        "R"
    } else if is_vector_arith(encoding) || is_vector_load(encoding) || is_vector_store(encoding) {
        "V"
    } else if fields == ["rd", "rs1", "rs2", "rm"] || fields == ["rd", "rs1", "rm"] {
        "R_rm"
    } else if fields == ["rd", "rs1", "rs2", "rs3", "rm"] {
        "R4_rm"
    } else if fields == ["rd", "rs1", "aq", "rl"] || fields == ["rd", "rs1", "rs2", "aq", "rl"] {
        "A"
    } else if fields == ["rd", "rs1", "imm12"]
        || fields == ["rd", "rs1", "shamtd"]
        || fields == ["rd", "rs1", "shamtw"]
        || fields == ["rd", "csr", "zimm5"]
        || fields == ["rd", "rs1", "csr"]
    {
        "I"
    } else if fields == ["imm12hi", "rs1", "rs2", "imm12lo"] {
        "S"
    } else if fields == ["bimm12hi", "rs1", "rs2", "bimm12lo"] {
        "B"
    } else if fields == ["rd", "imm20"] {
        "U"
    } else if fields == ["imm"] {
        "J"
    } else if ["fence", "fence_i"].contains(&name) {
        "I"
    } else if name == "jal" {
        "J"
    } else if fields.is_empty() {
        "None"
    }
    // RVC
    else if ["c_add", "c_jr", "c_jalr", "c_mv", "c_nop"].contains(&name) {
        "CR"
    } else if [
        "c_addi",
        "c_addiw",
        "c_addi16sp",
        "c_ldsp",
        "c_lwsp",
        "c_li",
        "c_lui",
        "c_slli",
        "c_flwsp",
        "c_fldsp",
    ]
    .contains(&name)
    {
        "CI"
    } else if name == "c_addi4spn" {
        "CIW"
    } else if ["c_and", "c_or", "c_xor", "c_sub", "c_addw", "c_subw"].contains(&name) {
        "CA"
    } else if ["c_beqz", "c_bnez", "c_srai", "c_srli", "c_andi"].contains(&name) {
        "CB"
    } else if ["c_j", "c_jal"].contains(&name) {
        "CJ"
    } else if ["c_lw", "c_ld", "c_flw", "c_fld"].contains(&name) {
        "CL"
    } else if ["c_sw", "c_fsw", "c_fsd", "c_sd"].contains(&name) {
        "CS"
    } else if ["c_swsp", "c_sdsp", "c_fswsp", "c_fsdsp"].contains(&name) {
        "CSS"
    } else {
        panic!(
            "Unknown instruction format for {} with fields: {}",
            name,
            fields.join(", ")
        );
    }
}

fn parse_instr<'a>(
    isa_dict: &mut HashMap<&'a str, Vec<String>>,
    ext_to_name: &HashMap<&str, &'a str>,
    json_path: &PathBuf,
) {
    let target_ext = ext_to_name.keys().collect::<Vec<_>>();

    let data = fs::read_to_string(&json_path).expect("Failed to read instr.json");
    let v: Value = serde_json::from_str(&data).expect("Invalid JSON");

    for (name, instr) in instr_object(&v, json_path) {
        let exts = instr_array(instr, name, "extension");

        if let Some(ext) = exts
            .iter()
            .map(|val| value_str(val, name, "extension"))
            .find(|e| target_ext.contains(&e))
        {
            let encoding = instr_str(instr, name, "encoding");

            if vector_ignore_instr(name, encoding) {
                continue;
            }

            let fields = instr_array(instr, name, "variable_fields")
                .iter()
                .map(|f| value_str(f, name, "variable_fields"))
                .collect::<Vec<_>>();

            let format = get_instr_type(fields.clone(), name, ext, encoding);
            let mask = hex_to_u64(instr_str(instr, name, "mask"));
            let key = hex_to_u64(instr_str(instr, name, "match"));

            let s = format!(
                "{} {{\n    format: InstrFormat::{},\n    mask: {},\n    key: {},\n}}",
                name.to_uppercase(),
                format,
                mask,
                key,
            );

            isa_dict
                .entry(ext_to_name.get(ext).unwrap())
                .or_default()
                .push(s);
        }
    }
}

fn get_commit_hash() -> String {
    if let Ok(hash) = env::var("GIT_HASH")
        && !hash.is_empty()
    {
        return hash;
    }

    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output();

    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        _ => String::from("unknown"),
    }
}

/// Generate the `define_riscv_isa!` macro
fn generate_isa_code(json_paths: &[PathBuf]) {
    let ext_to_name: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("rv_i", "RV32I");
        m.insert("rv_m", "RV32M");
        m.insert("rv64_i", "RV64I");
        m.insert("rv64_m", "RV64M");
        m.insert("rv_zicsr", "RVZicsr");
        m.insert("rv_system", "RVSystem");
        m.insert("rv_f", "RV32F");
        m.insert("rv64_f", "RV64F");
        m.insert("rv_d", "RV32D");
        m.insert("rv64_d", "RV64D");
        m.insert("rv_s", "RVS");
        m.insert("rv_a", "RV32A");
        m.insert("rv64_a", "RV64A");
        m.insert("rv_zifencei", "RVZifencei");
        m.insert("rv_v", "RVV");

        m.insert("rv_c", "RVC");
        m.insert("rv32_c", "RV32C");
        m.insert("rv32_c_f", "RV32C_F");
        m.insert("rv64_c", "RV64C");
        m.insert("rv_c_d", "RVC_D");

        // we defined illegal instruction (all zero) as an instruction for convenience,
        // kept separate from the auto-generated instr_dict.json
        m.insert("rv_illegal", "RVIllegal");

        #[cfg(feature = "custom-instr")]
        m.insert("rv_custom0", "RVCustom0");
        #[cfg(feature = "custom-instr")]
        m.insert("rv_custom1", "RVCustom1");
        m
    };

    let mut isa_dict: HashMap<&str, Vec<String>> = HashMap::new();
    for path in json_paths {
        parse_instr(&mut isa_dict, &ext_to_name, path);
    }

    let mut output = String::new();
    output.push_str("define_riscv_isa!(\n");
    output.push_str("RiscvInstr,\n");

    for (name, arr) in isa_dict.into_iter() {
        output.push_str(&format!(
            "{}, {}, {{\n",
            name,
            String::from("TABLE_") + &(*name).to_uppercase()
        ));
        for instr in arr {
            output.push_str(&format!("{},\n", instr));
        }
        output.push_str("},\n");
    }

    output.push_str(");\n");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("rvinstr_gen.rs");
    fs::write(&out_path, output).expect("Failed to write rvinstr_gen.rs");

    for path in json_paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    let json_paths = vec![
        PathBuf::from("./data/instr_dict.json"),
        PathBuf::from("./data/instr_dict_illegal.json"),
        #[cfg(feature = "custom-instr")]
        PathBuf::from("./data/instr_dict_custom.json"),
    ];

    generate_isa_code(&json_paths);

    // Expose commit hash to the crate via env! macro.
    println!("cargo:rerun-if-env-changed=GIT_HASH");
    println!("cargo:rustc-env=HERE_COMMIT_HASH={}", get_commit_hash());
}
