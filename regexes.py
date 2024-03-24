(\d+(\.\d+)?\s*\(\d+(\.\d+)?\)\s+){2,}   tables
(?:\d+\.\d+\s){2,}\d+\.\d+ consecutive triples of numbers
(?:\s|^)(\d+(\.\d+)?\s+){3,} two consecutive numbers separated by whitespace
(?:\d+\.\d+|\d+)(?:\s+(?:\d+\.\d+|\d+)){2,} used again two or more consecutive numbers (any type) separated by whitespace
(?:\s|^)(-?\d+(\.\d+)?\s+){3,}
(?:\b\d+(?:\.\d+)?(?: \(\d+(?:\.\d+)?%?\))(?:\s+|$)){2,} useda againtables with percentiles
 ?:\s+–?\d+(?:\.\d+)?(?:\s+\d+\s+\(\d+(?:\.\d+)?\))?\s*){3,} tabular data\b\d+\s+of\s+\d+\b
(?:\s+–?\d+(?:\.\d+)?(?:\s+\d+\s+\(\d+(?:\.\d+)?\))?\s*){2,}
 \b\d+\s+of\s+\d+\b



`\d+,\d+(?:,\d+)*\. ` -> `. `
`\.\d+,\d+(?:,\d+)*` -> `.`
`-\n` -> ``

`\[(\s*\d+(?:,\s*\d+)*\s*)\]` -> ``
`\[(\d+(?:,\d+)*)\]` -> ``
`\[(\d+)-(\d+)\]` -> ``
`\((\d+)-(\d+)\)\.` -> `.`
`\[(\d+)–(\d+)\]` -> ``
`\(\d{2}\)` -> ``
`\((\d{2})–(\d{2})\)`-> ``
`\.(\d{2})–(\d{2}) ` -> `. `
`(\d{2})–(\d{2})\. ` -> `.`
  
`[A-Z](\d{2}),(\d{2}) `
       
`[a-z](\d{2}) `
`[a-zA-Z]+\.([\d+,]+)-(\d+) `
`[a-zA-Z]+\.([\d+,]+)–(\d+) `
 `\s{2,}` -> ` ` 
`\[\d+([,-]\d+)+\]` -> ``
`\b\d{2},\d{2}(?:,\d{2})*\b ` -> ``
` \[\d+([,–]\d+)+\]` -> ``
`et al(\d+)` -> `et al.`
`\b[a-zA-Z]{5}\d+\b`
`([,]\d{2})\n` -> `,`
`([,]\d{2}) ` -> `, `
` . ` -> `. `
` , ` -> `,   `
`␣` -> `-alpha-`
`SarsCov` -> `Sars-Cov`
`metaa` -> `meta-a`
`^.{1,3}$` -> put . at the end of the previous line
`\b[A-Za-z]+,\d+\b` -> removal of refs
`·`
`–\n` -> `–`
`,\n` -> `,`
`;\n` -> `;`
`,,` -> `,`
`(\r?\n|\r){3,}` -> `\n\n`
`⋅` -> `.`
`�` -> `≥`
`⁄`-> `/`
`’’` -> 

`nmol / L` -> `nmol/L`
`nmol L 1` -> `nmol/L`
`nmol L-1` -> `nmol/L`
`nmol l−1` -> `nmol/L`

`^(?!.*(?:\.|\.")$).+` -> manually bring the line below, up.
`.’` -> `’.`
`.”` -> `”.`
`­ ` -> `` U+00AD is the Unicode code point for the soft hyphen (SHY) character

`	 `
앐
`ﬁ` -> `fi`
`ﬂ` -> `fl`
`ﬀ` -> `ff`
`ﬃ` -> `ffi`
`⌬` -> `Δ`
`␤` -> ``
`␮` -> μ
`⬃` -> `~`
•
• 
∙
►




 -> ≥
◦ -> °
쏝 -> <
 ҃ 
≤
≥
“
”
±
(?<!\S)\b\w{1,15},\w+(?!\S)
(?<=\d),(?=\d{3}(?!\d)) -> ``
(reviewed in Refs.(10,64,72)) -> ``
\((\d+)\) -> ``
`¼` -> `=`
` °C` -> `°C`
(\d+)\] -> remove all surrounding occurences
 \b[a-zA-Z]+\s*\d+,\d+(?:,\d+)*\.



 U+2424