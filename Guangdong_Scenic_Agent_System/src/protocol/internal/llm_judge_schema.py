{
  "$schema": "http://json-schema.org/draft/2020-12/schema",
  "title": "ScenicRouteJudge‑V2.2 LLM路由裁判内部契约",
  "description": "对内：硬匹配失败后LLM裁判输出，支持代词承接、allow_cross_city、spot_name_list多景区；禁止直接对外输出",
  "type": "object",
  "required": [
    "intent_type",
    "spot_name",
    "spot_name_list",
    "city",
    "scenic_level",
    "target_fields",
    "can_use_sql",
    "confidence_score",
    "reason",
    "allow_cross_city"
  ],
  "properties": {
    "intent_type": {
      "type": "string",
      "enum": ["spot_detail", "city_scenic_list", "rag_required"],
      "description": "意图类型：spot_detail=单景区详情反弹SQL；city_scenic_list=城市清单SQL；rag_required=走RAG"
    },
    "spot_name": {
      "type": "string",
      "description": "景区全称，识别不到填空字符串\"\"；intent_type=spot_detail时此字段不能为空",
      "default": ""
    },
    "spot_name_list": {
      "type": "array",
      "items": {"type":"string"},
      "description": "指代多个景区时填入；单景区/识别不到填[]；置信度>=95才采信",
      "default": []
    },
    "city": {
      "type": "string",
      "description": "广东省地级市；识别不到填空字符串\"\"；city_scenic_list时此字段不能为空",
      "default": ""
    },
    "scenic_level": {
      "type": "string",
      "enum": ["5A","4A","3A",""],
      "description": "景区星级，空字符串代表不限星级",
      "default": ""
    },
    "target_fields": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["ticket","open_time","address","telephone","level"]
      },
      "description": "当intent_type=spot_detail生效：用户想要查询哪些结构化字段；其他意图置为空数组[]",
      "default": []
    },
    "can_use_sql": {
      "type": "boolean",
      "description": "是否允许执行SQL；intent_type为rag_required时固定为false"
    },
    "confidence_score": {
      "type": "integer",
      "minimum": 0,
      "maximum": 100,
      "description": "置信分数，整数；SQL放行、spot_name_list采信阈值≥95"
    },
    "reason": {
      "type": "string",
      "description": "判断理由，仅日志调试，不参与业务逻辑"
    },
    "allow_cross_city": {
      "type": "boolean",
      "description": "是否允许跨城市周边推荐；true允许周边；false限定目标城市内；默认false"
    }
  },
  "additionalProperties": true,
  "allOf": [
    {
      "if": {
        "properties": {"intent_type": {"const": "spot_detail"}}
      },
      "then": {
        "properties": {"spot_name": {"minLength": 1}, "target_fields": {"minItems": 1}},
        "required": ["spot_name","target_fields"]
      }
    },
    {
      "if": {
        "properties": {"intent_type": {"const": "city_scenic_list"}}
      },
      "then": {
        "properties": {"city": {"minLength": 1}, "can_use_sql": {"const": true}},
        "required": ["city"]
      }
    },
    {
      "if": {
        "properties": {"intent_type": {"const": "rag_required"}}
      },
      "then": {
        "properties": {"can_use_sql": {"const": false}}
      }
    }
  ]
}
