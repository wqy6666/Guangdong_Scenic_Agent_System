{
  "judge_output_schema": {
    "can_use_sql": {
      "type": "boolean",
      "desc": "路由开关，true允许走SQL查询数据库，false走RAG向量检索"
    },
    "city": {
      "type": "string",
      "desc": "提取广东省地级市，识别不到填空字符串"
    },
    "scenic_level": {
      "type": "string",
      "desc": "景区星级，仅允许5A/4A/3A，识别不到填空字符串"
    },
    "confidence_score": {
      "type": "integer",
      "desc": "判断置信度，取值范围0‑100"
    },
    "reason": {
      "type": "string",
      "desc": "判断理由，仅用于日志调试，不参与路由分支逻辑"
    }
  },
  "threshold": {
    "sql_pass_score": 95,
    "min_score": 0,
    "max_score": 100
  },
  "sql_allow_city_list": [
    "广州市",
    "深圳市",
    "珠海市",
    "汕头市",
    "佛山市",
    "韶关市",
    "湛江市",
    "肇庆市",
    "江门市",
    "茂名市",
    "惠州市",
    "梅州市",
    "汕尾市",
    "河源市",
    "阳江市",
    "清远市",
    "东莞市",
    "中山市",
    "潮州市",
    "揭阳市",
    "云浮市"
  ],
  "sql_allow_level_list": ["5A", "4A", "3A"],
  "keyword_blacklist": ["门票", "票价", "收费", "开放时间", "预约", "交通", "停车", "营业时间"],
  "prompt_template": "你是景区查询路由判断器，严格输出JSON，禁止输出任何JSON之外的文字、注释、markdown标记。\n输出字段要求：\n1. can_use_sql：布尔值；只有问题是查询【城市+星级】类景区清单时才可以为true；如果命中黑名单关键词必须设置为false。\n2. city：只从sql_allow_city_list列表提取城市，识别不出返回空字符串。\n3. scenic_level：只允许5A/4A/3A，识别不出返回空字符串。\n4. confidence_score：整数0‑100，只有信息明确才给高分；信息模糊、残缺要压低分数。\n5. reason：写明判断依据，用于日志排查。\n只有confidence_score大于等于{{sql_pass_score}}并且can_use_sql为true，才放行SQL分支。\n黑名单关键词：{{keyword_blacklist}}\n用户问题：{{user_query}}"
}
