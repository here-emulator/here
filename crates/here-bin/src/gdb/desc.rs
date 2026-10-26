use gdbstub_arch::riscv::reg;
use std::fmt::Write;
use std::sync::OnceLock;

const XML_PREFIX: &str = r#"
<?xml version="1.0"?>
<!DOCTYPE feature SYSTEM "gdb-target.dtd">
<target>
    <architecture>riscv</architecture>
"#;

const CPU_FEATURE: &str = r#"
<feature name="org.gnu.gdb.riscv.cpu">
    <reg name="zero" bitsize="64" type="int" regnum="0"/>
    <reg name="ra" bitsize="64" type="code_ptr"/>
    <reg name="sp" bitsize="64" type="data_ptr"/>
    <reg name="gp" bitsize="64" type="data_ptr"/>
    <reg name="tp" bitsize="64" type="data_ptr"/>
    <reg name="t0" bitsize="64" type="int"/>
    <reg name="t1" bitsize="64" type="int"/>
    <reg name="t2" bitsize="64" type="int"/>
    <reg name="fp" bitsize="64" type="data_ptr"/>
    <reg name="s1" bitsize="64" type="int"/>
    <reg name="a0" bitsize="64" type="int"/>
    <reg name="a1" bitsize="64" type="int"/>
    <reg name="a2" bitsize="64" type="int"/>
    <reg name="a3" bitsize="64" type="int"/>
    <reg name="a4" bitsize="64" type="int"/>
    <reg name="a5" bitsize="64" type="int"/>
    <reg name="a6" bitsize="64" type="int"/>
    <reg name="a7" bitsize="64" type="int"/>
    <reg name="s2" bitsize="64" type="int"/>
    <reg name="s3" bitsize="64" type="int"/>
    <reg name="s4" bitsize="64" type="int"/>
    <reg name="s5" bitsize="64" type="int"/>
    <reg name="s6" bitsize="64" type="int"/>
    <reg name="s7" bitsize="64" type="int"/>
    <reg name="s8" bitsize="64" type="int"/>
    <reg name="s9" bitsize="64" type="int"/>
    <reg name="s10" bitsize="64" type="int"/>
    <reg name="s11" bitsize="64" type="int"/>
    <reg name="t3" bitsize="64" type="int"/>
    <reg name="t4" bitsize="64" type="int"/>
    <reg name="t5" bitsize="64" type="int"/>
    <reg name="t6" bitsize="64" type="int"/>
    <reg name="pc" bitsize="64" type="code_ptr"/>
</feature>
"#;

const FLOAT_F_FEATURE: &str = r#"
<feature name="org.gnu.gdb.riscv.fpu">
    <reg name="ft0" bitsize="32" type="ieee_single" regnum="33"/>
    <reg name="ft1" bitsize="32" type="ieee_single"/>
    <reg name="ft2" bitsize="32" type="ieee_single"/>
    <reg name="ft3" bitsize="32" type="ieee_single"/>
    <reg name="ft4" bitsize="32" type="ieee_single"/>
    <reg name="ft5" bitsize="32" type="ieee_single"/>
    <reg name="ft6" bitsize="32" type="ieee_single"/>
    <reg name="ft7" bitsize="32" type="ieee_single"/>
    <reg name="fs0" bitsize="32" type="ieee_single"/>
    <reg name="fs1" bitsize="32" type="ieee_single"/>
    <reg name="fa0" bitsize="32" type="ieee_single"/>
    <reg name="fa1" bitsize="32" type="ieee_single"/>
    <reg name="fa2" bitsize="32" type="ieee_single"/>
    <reg name="fa3" bitsize="32" type="ieee_single"/>
    <reg name="fa4" bitsize="32" type="ieee_single"/>
    <reg name="fa5" bitsize="32" type="ieee_single"/>
    <reg name="fa6" bitsize="32" type="ieee_single"/>
    <reg name="fa7" bitsize="32" type="ieee_single"/>
    <reg name="fs2" bitsize="32" type="ieee_single"/>
    <reg name="fs3" bitsize="32" type="ieee_single"/>
    <reg name="fs4" bitsize="32" type="ieee_single"/>
    <reg name="fs5" bitsize="32" type="ieee_single"/>
    <reg name="fs6" bitsize="32" type="ieee_single"/>
    <reg name="fs7" bitsize="32" type="ieee_single"/>
    <reg name="fs8" bitsize="32" type="ieee_single"/>
    <reg name="fs9" bitsize="32" type="ieee_single"/>
    <reg name="fs10" bitsize="32" type="ieee_single"/>
    <reg name="fs11" bitsize="32" type="ieee_single"/>
    <reg name="ft8" bitsize="32" type="ieee_single"/>
    <reg name="ft9" bitsize="32" type="ieee_single"/>
    <reg name="ft10" bitsize="32" type="ieee_single"/>
    <reg name="ft11" bitsize="32" type="ieee_single"/>

    <reg name="fcsr" bitsize="32" type="int" regnum="68"/>
</feature>
"#;

const FLOAT_D_FEATURE: &str = r#"
<feature name="org.gnu.gdb.riscv.fpu">
    <union id="riscv_double">
        <field name="float" type="ieee_single"/>
        <field name="double" type="ieee_double"/>
    </union>

    <reg name="ft0" bitsize="64" type="riscv_double" regnum="33"/>
    <reg name="ft1" bitsize="64" type="riscv_double"/>
    <reg name="ft2" bitsize="64" type="riscv_double"/>
    <reg name="ft3" bitsize="64" type="riscv_double"/>
    <reg name="ft4" bitsize="64" type="riscv_double"/>
    <reg name="ft5" bitsize="64" type="riscv_double"/>
    <reg name="ft6" bitsize="64" type="riscv_double"/>
    <reg name="ft7" bitsize="64" type="riscv_double"/>
    <reg name="fs0" bitsize="64" type="riscv_double"/>
    <reg name="fs1" bitsize="64" type="riscv_double"/>
    <reg name="fa0" bitsize="64" type="riscv_double"/>
    <reg name="fa1" bitsize="64" type="riscv_double"/>
    <reg name="fa2" bitsize="64" type="riscv_double"/>
    <reg name="fa3" bitsize="64" type="riscv_double"/>
    <reg name="fa4" bitsize="64" type="riscv_double"/>
    <reg name="fa5" bitsize="64" type="riscv_double"/>
    <reg name="fa6" bitsize="64" type="riscv_double"/>
    <reg name="fa7" bitsize="64" type="riscv_double"/>
    <reg name="fs2" bitsize="64" type="riscv_double"/>
    <reg name="fs3" bitsize="64" type="riscv_double"/>
    <reg name="fs4" bitsize="64" type="riscv_double"/>
    <reg name="fs5" bitsize="64" type="riscv_double"/>
    <reg name="fs6" bitsize="64" type="riscv_double"/>
    <reg name="fs7" bitsize="64" type="riscv_double"/>
    <reg name="fs8" bitsize="64" type="riscv_double"/>
    <reg name="fs9" bitsize="64" type="riscv_double"/>
    <reg name="fs10" bitsize="64" type="riscv_double"/>
    <reg name="fs11" bitsize="64" type="riscv_double"/>
    <reg name="ft8" bitsize="64" type="riscv_double"/>
    <reg name="ft9" bitsize="64" type="riscv_double"/>
    <reg name="ft10" bitsize="64" type="riscv_double"/>
    <reg name="ft11" bitsize="64" type="riscv_double"/>

    <reg name="fcsr" bitsize="32" type="int" regnum="68"/>
</feature>
"#;

const XML_SUFFIX: &str = "</target>";

static TARGET_DESCRIPTION_XML: OnceLock<String> = OnceLock::new();

pub struct DescBuilder {
    csr_regs: Vec<(u64, String)>,
    float_feature: FloatFeature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FloatFeature {
    #[default]
    None,
    F,
    D,
}

impl DescBuilder {
    pub fn new() -> Self {
        Self {
            csr_regs: Vec::new(),
            float_feature: FloatFeature::None,
        }
    }

    pub fn with_csrs<I, S>(csr_regs: I) -> Self
    where
        I: IntoIterator<Item = (u64, S)>,
        S: Into<String>,
    {
        let csr_regs = csr_regs
            .into_iter()
            .map(|(addr, name)| (addr, name.into()))
            .collect::<Vec<_>>();

        Self {
            csr_regs,
            float_feature: FloatFeature::None,
        }
    }

    pub fn add_csr(mut self, addr: u64, name: impl Into<String>) -> Self {
        self.csr_regs.push((addr, name.into()));
        self
    }

    pub fn with_f(mut self) -> Self {
        self.float_feature = FloatFeature::F;
        self
    }

    pub fn with_d(mut self) -> Self {
        self.float_feature = FloatFeature::D;
        self
    }

    pub fn build(mut self) -> String {
        self.csr_regs.sort_unstable_by_key(|(addr, _)| *addr);

        let mut xml = String::new();
        xml.push_str(XML_PREFIX);
        xml.push_str(CPU_FEATURE);

        match self.float_feature {
            FloatFeature::None => {}
            FloatFeature::F => xml.push_str(FLOAT_F_FEATURE),
            FloatFeature::D => xml.push_str(FLOAT_D_FEATURE),
        }

        xml.push_str("<feature name=\"org.gnu.gdb.riscv.csr\">\n");
        for (addr, name) in self.csr_regs {
            let regnum = 65 + addr;
            let _ = writeln!(
                xml,
                r#"    <reg name="{name}" bitsize="64" type="int" regnum="{regnum}"/>"#,
            );
        }
        xml.push_str("    </feature>\n");
        xml.push_str(XML_SUFFIX);
        xml
    }
}

pub fn init_target_desc_xml(builder: DescBuilder) {
    let _ = TARGET_DESCRIPTION_XML.get_or_init(|| builder.build());
}

pub fn init_target_desc_xml_raw(xml: String) {
    let _ = TARGET_DESCRIPTION_XML.get_or_init(|| xml);
}

pub enum Riscv64 {}

impl gdbstub::arch::Arch for Riscv64 {
    type Usize = u64;
    type Registers = reg::RiscvCoreRegs<u64>;
    type BreakpointKind = usize;
    type RegId = reg::id::RiscvRegId<u64>;

    fn target_description_xml() -> Option<&'static str> {
        TARGET_DESCRIPTION_XML.get().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::DescBuilder;

    #[test]
    fn build_sorts_csrs_and_skips_fpu_by_default() {
        let xml = DescBuilder::with_csrs([(0x305, "mtvec"), (0x300, "mstatus")]).build();

        assert!(xml.contains("<feature name=\"org.gnu.gdb.riscv.cpu\">"));
        assert!(!xml.contains("<feature name=\"org.gnu.gdb.riscv.fpu\">"));
        assert!(xml.contains("<reg name=\"mstatus\" bitsize=\"64\" type=\"int\" regnum=\"833\"/>"));
        assert!(xml.contains("<reg name=\"mtvec\" bitsize=\"64\" type=\"int\" regnum=\"838\"/>"));
        assert!(xml.find("mstatus").unwrap() < xml.find("mtvec").unwrap());
    }

    #[test]
    fn build_includes_f_extension() {
        let xml = DescBuilder::with_csrs([(0x003, "fcsr")]).with_f().build();

        assert!(xml.contains("<feature name=\"org.gnu.gdb.riscv.fpu\">"));
        assert!(
            xml.contains("<reg name=\"ft0\" bitsize=\"32\" type=\"ieee_single\" regnum=\"33\"/>")
        );
        assert!(xml.contains("<reg name=\"fcsr\" bitsize=\"32\" type=\"int\" regnum=\"68\"/>"));
    }

    #[test]
    fn build_includes_d_extension() {
        let xml = DescBuilder::with_csrs([(0x003, "fcsr")]).with_d().build();

        assert!(xml.contains("<feature name=\"org.gnu.gdb.riscv.fpu\">"));
        assert!(xml.contains("<union id=\"riscv_double\">"));
        assert!(xml.contains("<reg name=\"fcsr\" bitsize=\"32\" type=\"int\" regnum=\"68\"/>"));
    }
}
