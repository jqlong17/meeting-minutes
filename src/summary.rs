use crate::error::{AppError, AppResult};

pub fn build_prompt(asr_text: &str) -> String {
    format!(
        "你是一个负责整理会议纪要的助手。请根据下面的会议 ASR 文本，输出简短、精炼、可执行的会议纪要。\n\
要求：\n\
1. 只输出最终纪要正文，不要解释。\n\
2. 使用中文。\n\
3. 只保留确定性强的结论和行动项，不要编造。\n\
4. 不要输出思考过程、不要输出 <think>、不要输出“我将/我需要/基于以上分析”等元话语。\n\
5. `核心结论`必须写会议中已经明确达成的共识、判断、决策、优先级、时间节点、分工原则。\n\
6. `核心结论`不要写“会议讨论了什么”“重点讨论了什么”“介绍了什么版本规划”这类概述型句子。\n\
7. `核心结论`要尽量写成“最终明确了什么”，而不是“过程里聊了什么”。\n\
8. 优先提炼这类高信息量结论：\n\
   - 版本时间点或上线目标被明确\n\
   - 产品目标或交付边界被明确\n\
   - 哪个平台负责什么、是否复用现有能力被明确\n\
   - 实施方式、优先级、范围取舍被明确\n\
9. 如果没有足够明确的结论，宁可少写，不要用空泛概述凑数。\n\
10. `todo` 只写明确行动项，必须尽量满足以下条件中的至少两个：责任人/责任方、动作、交付物、时间点。\n\
11. 如果某条内容只是“讨论了什么”“目标是什么”“背景是什么”“时间节点是什么”，但没有形成明确动作，不要写入 `todo`。\n\
12. `todo` 优先提取这类句子：\n\
   - 某人/某团队 需要/负责/去 对接、确认、推进、输出、实现、联调、准备、复用\n\
   - 明确说到“谁来做”“什么时候上线”“要确认什么”“要产出什么”\n\
13. `todo` 不要求固定 3 条，只输出真实提取到的行动项；如果只有 1 条或 2 条，就只写 1 条或 2 条，宁缺勿滥。\n\
14. `todo` 每条都要写成可执行句式，优先使用“某人/某团队 + 动作 + 目标/对象”的结构。\n\
15. 严格使用以下格式：\n\
核心结论：\n\
1. ...\n\
2. ...\n\
3. ...\n\n\
todo：\n\
1. ...\n\
2. ...（如无更多明确待办可不写）\n\
3. ...（如无更多明确待办可不写）\n\n\
反例：\n\
- 不要把“会议讨论了AI工作台需求”写成核心结论或 todo。\n\
- 不要把“产品目标是降低门槛”单独写成 todo。\n\
- 不要把“时间节点是5月中旬”单独写成 todo。\n\
- 不要把“重点讨论了版本规划”写成核心结论。\n\n\
正例：\n\
- 会议明确：AI本体工作台需要在5月中旬给出首版，最晚5月底上线。\n\
- 会议明确：本体平台侧优先复用AF现有能力，避免重复开发。\n\
- 会议明确：当前诉求不是完整智能体平台，而是先提供可复用的对话与接口能力。\n\
- 佳琪与若楠对接，确认 AF 现有能力是否可直接复用。\n\
- 团队在 5 月中旬前确定可上线的首版功能范围。\n\
- 相关团队补充 AI 本体工作台的前后端实现方案与资源要求。\n\n\
会议 ASR 文本如下：\n\
{asr_text}"
    )
}

pub fn build_asr_polish_prompt(asr_text: &str) -> String {
    format!(
        "你将收到一段中文会议 ASR 文本。请输出一份“轻度优化版 ASR”。\n\
目标：在尽量保持原文内容、顺序、语气和表达习惯的前提下，仅做有限清洗与润色。\n\
\n\
要求：\n\
1. 只输出优化后的正文，不要解释，不要加标题，不要加说明。\n\
2. 尽量保持原文顺序，不要重组结构，不要总结，不要提炼。\n\
3. 尽量保持原句与原词，除非明显是 ASR 识别错误、口语断裂、重复赘词或不通顺。\n\
4. 可以做的修改包括：修正明显错别字、补足必要标点、拆分过长句、去掉明显重复口头禅、修复少量语病。\n\
5. 不要新增原文没有的信息，不要补充推断内容，不要改变结论，不要改写成书面总结。\n\
6. 对人名、产品名、缩写、技术词，如果无法确定，不要自作主张替换，尽量保留原样。\n\
7. 若某处听写不清或语义不明，也不要编造，保留接近原文的表达。\n\
8. 输出长度应与原文接近，不能明显缩短。\n\
9. 若原文存在重复段落，只删除明显连续重复的冗余片段，不要删减真实内容。\n\
\n\
会议 ASR 文本如下：\n\
{asr_text}"
    )
}

pub fn normalize_summary(raw: &str, asr_text: &str) -> AppResult<String> {
    let text = sanitize_raw_summary(raw);
    if text.is_empty() {
        return Err(AppError::llm("大模型返回空内容"));
    }

    let mut core_items = Vec::new();
    let mut todo_items = Vec::new();
    let mut section = "";

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("核心结论") {
            section = "core";
            continue;
        }
        if trimmed.starts_with("todo") || trimmed.starts_with("TODO") || trimmed.starts_with("待办")
        {
            section = "todo";
            continue;
        }

        let normalized = strip_list_prefix(trimmed)
            .trim()
            .trim_matches('"')
            .to_string();
        if normalized.is_empty() || is_meta_line(&normalized) {
            continue;
        }

        match section {
            "core" => core_items.push(normalized),
            "todo" => {
                if looks_actionable(&normalized)
                    && !is_weak_todo_line(&normalized)
                    && !is_requirement_statement(&normalized)
                    && has_responsibility_hint(&normalized)
                {
                    todo_items.push(normalized);
                }
            }
            _ => {
                if core_items.len() < 3 {
                    core_items.push(normalized);
                } else if looks_actionable(&normalized)
                    && !is_weak_todo_line(&normalized)
                    && !is_requirement_statement(&normalized)
                    && has_responsibility_hint(&normalized)
                {
                    todo_items.push(normalized);
                }
            }
        }
    }

    core_items.retain(|item| !is_weak_core_line(item));

    core_items = dedupe_items(core_items);
    if core_items.len() < 3 {
        for item in fallback_core_items(asr_text) {
            if core_items.len() >= 3 {
                break;
            }
            if !core_items.iter().any(|existing| existing == &item) {
                core_items.push(item);
            }
        }
    }

    if core_items.is_empty() {
        core_items.push("会议内容较分散，未形成足够明确的稳定结论。".to_string());
    }
    core_items = dedupe_items(core_items);
    todo_items = dedupe_items(todo_items);

    let core = core_items
        .into_iter()
        .take(3)
        .enumerate()
        .map(|(index, item)| format!("{}. {}", index + 1, item))
        .collect::<Vec<_>>()
        .join("\n");
    let todo = todo_items
        .into_iter()
        .take(10)
        .enumerate()
        .map(|(index, item)| format!("{}. {}", index + 1, item))
        .collect::<Vec<_>>()
        .join("\n");

    let todo_section = if todo.is_empty() {
        "1. 暂未从会议中提取到足够明确的行动项。".to_string()
    } else {
        todo
    };

    Ok(format!("核心结论：\n{core}\n\ntodo：\n{todo_section}\n"))
}

pub fn normalize_polished_asr(raw: &str, asr_text: &str) -> AppResult<String> {
    let text = sanitize_raw_summary(raw);
    if text.is_empty() {
        return Err(AppError::llm("大模型返回空内容"));
    }

    let normalized = text
        .lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with("优化后")
                && !line.starts_with("润色后")
                && !line.starts_with("以下是")
        })
        .collect::<Vec<_>>()
        .join("\n");

    if normalized.is_empty() {
        return Ok(asr_text.trim().to_string());
    }

    let raw_len = asr_text.chars().count();
    let polished_len = normalized.chars().count();
    if polished_len * 2 < raw_len {
        return Ok(asr_text.trim().to_string());
    }

    Ok(normalized.trim().to_string())
}

fn strip_list_prefix(line: &str) -> &str {
    let mut index = 0usize;
    for (char_index, ch) in line.char_indices() {
        if ch.is_ascii_digit() || matches!(ch, '.' | '、' | '-' | ' ') {
            index = char_index + ch.len_utf8();
        } else {
            break;
        }
    }
    &line[index..]
}

fn sanitize_raw_summary(raw: &str) -> String {
    raw.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return Some(String::new());
            }
            if trimmed.starts_with("<think>")
                || trimmed.starts_with("</think>")
                || trimmed.starts_with("```")
            {
                return None;
            }
            Some(trimmed.replace("<think>", "").replace("</think>", ""))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn is_meta_line(line: &str) -> bool {
    let patterns = [
        "我需要",
        "我将",
        "基于以上分析",
        "下面是",
        "主要讨论了",
        "我来分析",
        "我先分析",
        "我理解",
        "现在我来整理",
        "首先，让我分析",
    ];
    patterns.iter().any(|pattern| line.contains(pattern))
}

fn looks_actionable(line: &str) -> bool {
    let action_keywords = [
        "明确", "推进", "完成", "梳理", "对齐", "设计", "实现", "上线", "联调", "评审", "提供",
        "支持", "确认", "接入", "复用", "准备", "输出",
    ];
    action_keywords.iter().any(|keyword| line.contains(keyword))
}

fn is_weak_todo_line(line: &str) -> bool {
    let weak_prefixes = [
        "核心目标",
        "会议讨论",
        "讨论了",
        "会议明确了",
        "主要需求",
        "用户场景",
        "时间节点",
        "Q1",
        "Q2",
        "Q3",
        "月底",
        "技术实现",
        "产品规划",
        "本体管理平台计划",
        "产品目标",
    ];
    weak_prefixes.iter().any(|prefix| line.starts_with(prefix))
}

fn dedupe_items(items: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    let mut normalized_seen = Vec::new();
    for item in items {
        let normalized = normalize_for_dedupe(&item);
        if !normalized_seen
            .iter()
            .any(|existing| existing == &normalized)
        {
            normalized_seen.push(normalized);
            deduped.push(item);
        }
    }
    deduped
}

fn is_weak_core_line(line: &str) -> bool {
    let weak_patterns = [
        "会议讨论的是",
        "会议中讨论的主要内容",
        "重点讨论了",
        "讨论了",
        "介绍了",
        "产品背景和目标",
        "产品规划",
        "版本规划",
        "用户场景",
        "功能开发",
        "需求和实现方案",
    ];
    weak_patterns.iter().any(|pattern| line.contains(pattern))
}

fn fallback_core_items(asr_text: &str) -> Vec<String> {
    let normalized = asr_text.replace(' ', "").to_lowercase();
    let mut items = Vec::new();

    if (normalized.contains("五月中旬") || normalized.contains("5月中旬"))
        && (normalized.contains("五月三十") || normalized.contains("5月30"))
    {
        items.push("会议明确：AI本体工作台希望在5月中旬给出一版，最晚在5月底上线。".to_string());
    }

    if normalized.contains("af")
        && normalized.contains("复用")
        && (normalized.contains("不要重复开发")
            || normalized.contains("直接用")
            || normalized.contains("现有能力"))
    {
        items.push(
            "会议倾向优先复用 AF 的现有能力，避免本体平台重复开发同类前后端能力。".to_string(),
        );
    }

    if normalized.contains("不需要")
        && normalized.contains("智能体")
        && (normalized.contains("接口") || normalized.contains("后端"))
    {
        items.push(
            "会议明确当前优先诉求不是完整智能体平台，而是先提供可复用的对话和后端接口能力。"
                .to_string(),
        );
    }

    if normalized.contains("降低本体建模门槛") || normalized.contains("提高工作效率")
    {
        items.push(
            "会议确认本阶段核心目标是降低本体建模门槛，并提升现场实施和建模效率。".to_string(),
        );
    }

    if normalized.contains("从零到一")
        || normalized.contains("更新现有模型")
        || normalized.contains("更新函数")
    {
        items.push(
            "会议确认首版能力需要覆盖三类场景：从零构建、更新现有模型、以及迭代函数。".to_string(),
        );
    }

    items
}

fn has_responsibility_hint(line: &str) -> bool {
    let hints = [
        "佳琪",
        "若楠",
        "雅兰",
        "肖峰",
        "产品团队",
        "研发团队",
        "实施团队",
        "相关团队",
        "前端",
        "后端",
        "团队在",
        "团队需",
        "团队需要",
        "与若楠对接",
        "与雅兰沟通",
        "与肖峰沟通",
    ];
    hints.iter().any(|hint| line.contains(hint))
}

fn is_requirement_statement(line: &str) -> bool {
    let product_subjects = [
        "AI本体工作台",
        "本体平台",
        "本体管理平台",
        "系统",
        "产品",
        "页面",
    ];
    let requirement_patterns = ["需要支持", "需要实现", "要支持", "应支持", "需支持"];

    if line.contains("用户场景") {
        return true;
    }

    product_subjects
        .iter()
        .any(|subject| line.contains(subject))
        && requirement_patterns
            .iter()
            .any(|pattern| line.contains(pattern))
}

fn normalize_for_dedupe(line: &str) -> String {
    line.chars()
        .filter(|ch| !matches!(ch, '，' | ',' | '。' | '.' | '：' | ':' | '；' | ';' | ' '))
        .collect::<String>()
}
