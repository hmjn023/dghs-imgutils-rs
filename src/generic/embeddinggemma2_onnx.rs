//! Minimal lossless protobuf editing for the hash-validated Q8 graphs.
use super::{GenericError, invalid};

#[derive(Clone)]
struct Field {
    tag: u64,
    wire: u64,
    value: Vec<u8>,
}
fn varint(input: &mut &[u8]) -> Result<u64, GenericError> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let (&byte, rest) = input
            .split_first()
            .ok_or_else(|| invalid("truncated protobuf"))?;
        *input = rest;
        if shift == 63 && byte > 1 {
            return Err(invalid("overflowing protobuf varint"));
        }
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    Err(invalid("invalid protobuf varint"))
}
fn put(mut value: u64, out: &mut Vec<u8>) {
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn parse(mut data: &[u8]) -> Result<Vec<Field>, GenericError> {
    let mut fields = Vec::new();
    while !data.is_empty() {
        let key = varint(&mut data)?;
        let wire = key & 7;
        if key >> 3 == 0 {
            return Err(invalid("invalid protobuf field"));
        }
        let size = match wire {
            0 => {
                let original = data;
                varint(&mut data)?;
                let n = original.len() - data.len();
                data = original;
                n
            }
            1 => 8,
            2 => usize::try_from(varint(&mut data)?).map_err(|_| invalid("protobuf too large"))?,
            5 => 4,
            _ => return Err(invalid("unsupported protobuf wire type")),
        };
        if size > data.len() {
            return Err(invalid("truncated protobuf field"));
        }
        fields.push(Field {
            tag: key >> 3,
            wire,
            value: data[..size].to_vec(),
        });
        data = &data[size..];
    }
    Ok(fields)
}
fn encode(fields: &[Field]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in fields {
        put(f.tag << 3 | f.wire, &mut out);
        if f.wire == 2 {
            put(f.value.len() as u64, &mut out);
        }
        out.extend_from_slice(&f.value);
    }
    out
}
fn bytes(tag: u64, value: impl AsRef<[u8]>) -> Field {
    Field {
        tag,
        wire: 2,
        value: value.as_ref().to_vec(),
    }
}
fn int(tag: u64, value: u64) -> Field {
    let mut v = Vec::new();
    put(value, &mut v);
    Field {
        tag,
        wire: 0,
        value: v,
    }
}
fn value(fields: &[Field], tag: u64) -> &[u8] {
    fields
        .iter()
        .find(|f| f.tag == tag)
        .map_or(&[], |f| f.value.as_slice())
}
fn attr(name: &str, v: Field, kind: u64) -> Field {
    bytes(5, encode(&[bytes(1, name), v, int(20, kind)]))
}
fn node(op: &str, inputs: &[&str], output: &str) -> Field {
    let mut f: Vec<_> = inputs.iter().map(|s| bytes(1, s)).collect();
    f.extend([bytes(2, output), bytes(4, op)]);
    bytes(1, encode(&f))
}
/// Makes RoPE positions explicit while retaining all unknown protobuf fields.
pub(super) fn patch(data: &[u8], shared: bool) -> Result<Vec<u8>, GenericError> {
    if !shared {
        return Ok(data.to_vec());
    }
    let mut model = parse(data)?;
    let graph = model
        .iter_mut()
        .find(|f| f.tag == 7)
        .ok_or_else(|| invalid("missing graph"))?;
    let mut fields = parse(&graph.value)?;
    let mut count = 0;
    for f in fields.iter_mut().filter(|f| f.tag == 1) {
        let mut n = parse(&f.value)?;
        if value(&n, 4) == b"RotaryEmbedding" && value(&n, 7) == b"com.microsoft" {
            let position = n
                .iter_mut()
                .filter(|f| f.tag == 1)
                .nth(1)
                .ok_or_else(|| invalid("missing RoPE position"))?;
            if position.value != b"/model/constants/INT64/[0]" {
                return Err(invalid("unexpected RoPE positions"));
            }
            position.value = b"dghs_rotary_position_ids".to_vec();
            f.value = encode(&n);
            count += 1;
        }
    }
    if count != 48 {
        return Err(invalid("unexpected RoPE node count"));
    }
    let mut cast = parse(
        &node(
            "Cast",
            &["/model/positions/Range/output_0"],
            "dghs_rotary_range",
        )
        .value,
    )?;
    cast.push(attr("to", int(3, 7), 2));
    // ONNX permits arbitrary node order at import, but keep dependencies ordered.
    let index = fields
        .iter()
        .position(|f| {
            f.tag == 1
                && parse(&f.value).is_ok_and(|n| {
                    value(&n, 4) == b"Range" && value(&n, 2) == b"/model/positions/Range/output_0"
                })
        })
        .ok_or_else(|| invalid("missing position Range"))?;
    fields.splice(
        index + 1..index + 1,
        [
            bytes(1, encode(&cast)),
            node(
                "Unsqueeze",
                &["dghs_rotary_range", "/model/constants/INT64/[0]"],
                "dghs_rotary_position_ids",
            ),
        ],
    );
    graph.value = encode(&fields);
    Ok(encode(&model))
}
/// Wraps an exported native GPU blob without changing model input/output order.
pub(super) fn wrapper(data: &[u8], blob: &str, sdk_version: &str) -> Result<Vec<u8>, GenericError> {
    let model = parse(data)?;
    let graph = parse(value(&model, 7))?;
    let mut node = vec![
        bytes(3, "OpenVINO"),
        bytes(4, "EPContext"),
        bytes(7, "com.microsoft"),
    ];
    for (tag, nt) in [(11, 1), (12, 2)] {
        for f in graph.iter().filter(|f| f.tag == tag) {
            node.push(bytes(nt, value(&parse(&f.value)?, 1)));
        }
    }
    node.extend([
        attr("source", bytes(4, "OpenVINOExecutionProvider"), 3),
        attr("embed_mode", int(3, 0), 2),
        attr("ep_cache_context", bytes(4, blob), 3),
        attr("partition_name", bytes(4, "OpenVINO"), 3),
        attr("ep_sdk_version", bytes(4, sdk_version), 3),
    ]);
    let mut g = vec![bytes(1, encode(&node)), bytes(2, "OpenVINO")];
    g.extend(graph.into_iter().filter(|f| matches!(f.tag, 11 | 12)));
    let mut m: Vec<_> = model
        .into_iter()
        .filter(|f| matches!(f.tag, 1 | 8))
        .collect();
    m.push(bytes(7, encode(&g)));
    Ok(encode(&m))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_roundtrip_and_malformed() {
        let f = vec![bytes(7, b"payload"), int(1, 300)];
        let b = encode(&f);
        assert_eq!(encode(&parse(&b).unwrap()), b);
        for b in [&[0u8][..], &[10, 255][..], &[15][..]] {
            assert!(parse(b).is_err());
        }
        assert!(patch(&encode(&[bytes(7, [])]), true).is_err());
    }
}
