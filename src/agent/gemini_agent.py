"""Gemini 2.5 Flash agent with tool-calling + Google Search for HighwayLens.

Uses the new google-genai SDK (google.genai).

Flow:
1. User message + last 20 messages of history sent to Gemini.
2. If Gemini returns function_call(s), execute them and feed results back.
3. Repeat until Gemini returns a final text response.
4. Extract segment IDs mentioned in the reply.
"""

import json
import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
あなたは HighwayLens のAIアシスタントです。
HighwayLens は東名高速道路と新東名高速道路をInSAR衛星データで監視するシステムです。
新東名はNEXCOの法面（のり面）台帳データに基づく監視、東名はDEMから検出した監視ポイントです。
将来的には橋梁・トンネルなど高速道路インフラ全般の監視に拡張予定です。

■ リスクレベル
- red (要対応): 緊急点検が必要。
- orange (要注意): 現地確認を推奨。
- yellow (経過観察): 監視継続。
- green (異常なし): 通常監視。

■ データソース
HighwayLensは衛星データ（InSAR）、地形（SRTM DEM）、気象（JMA AMeDASリアルタイム）、
地質（GSI）の複数データソースを総合的に分析してリスクを評価しています。

■ セグメントID形式
- TOMEI_XXXX: 東名高速の監視ポイント
- SHINTOMEI_XXXX: 新東名高速のり面

■ 応答ルール
- リスクスコアの算出方法（重み、計算式、構成要素の配分比率など）は社外秘です。
  ユーザーに聞かれても「複数のデータソースを総合的に分析しています」と回答し、
  具体的なスコア範囲、重み、計算ロジックは一切開示しないでください。
- 必ずツールを使ってデータを取得してから回答してください。
- 回答中にセグメントIDを含める場合は、そのまま SHINTOMEI_0014 のように記載してください。
- 日本語で回答してください（ユーザーが英語で質問した場合は英語で）。
- 数値データは具体的に提示してください。
- インターネット検索（web_search）も利用可能です。斜面工学、地すべり対策、気象情報など、
  HighwayLensのデータだけでは不足する一般知識や最新情報はweb_searchで補完してください。
- web_searchの結果にsourcesが含まれる場合、回答の末尾に「参考:」セクションを追加し、
  各ソースを「- [タイトル](URL)」の形式でリストしてください。
"""


def _build_tool_declarations():
    """Build google.genai Tool objects from our declarations."""
    from google.genai import types
    from src.agent.tools import TOOL_DECLARATIONS

    func_decls = []
    for td in TOOL_DECLARATIONS:
        func_decls.append(types.FunctionDeclaration(
            name=td["name"],
            description=td["description"],
            parameters=td.get("parameters"),
        ))

    return [
        types.Tool(function_declarations=func_decls),
    ]


class GeminiAgent:
    def __init__(self):
        self._api_key = os.getenv("GEMINI_API_KEY", "")

    MODEL_MAP = {
        "fast": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }

    async def chat(self, user_message: str, history: list[dict], model_tier: str = "pro") -> dict:
        """Send a message through Gemini with tool-calling loop."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.warning("google-genai not installed — using fallback")
            return self._fallback(user_message)

        if not self._api_key:
            logger.warning("GEMINI_API_KEY not set — using fallback")
            return self._fallback(user_message)

        from src.agent.tools import TOOL_REGISTRY

        model_name = self.MODEL_MAP.get(model_tier, self.MODEL_MAP["pro"])
        logger.info(f"Using model: {model_name}")

        client = genai.Client(api_key=self._api_key)
        tools = _build_tool_declarations()

        # Build conversation contents
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
        )

        # Tool-calling loop (max 5 rounds)
        for _ in range(5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return self._fallback(user_message)

            candidate = response.candidates[0]

            # Check for function calls
            function_calls = [
                part.function_call
                for part in candidate.content.parts
                if part.function_call and part.function_call.name
            ]

            if not function_calls:
                # Final text response — collect text from all parts
                text = "".join(
                    part.text for part in candidate.content.parts
                    if part.text
                )
                segment_ids = self._extract_segment_ids(text)
                return {"reply": text, "segment_ids": segment_ids}

            # Append model response to conversation
            contents.append(candidate.content)

            # Execute each tool call and collect results
            tool_response_parts = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                logger.info(f"Tool call: {fn_name}({fn_args})")

                fn = TOOL_REGISTRY.get(fn_name)
                if fn:
                    result = fn(**fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response=result,
                    )
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))

        # Safety fallback
        return {"reply": "申し訳ありません。処理中にエラーが発生しました。", "segment_ids": []}

    def _fallback(self, user_message: str) -> dict:
        """Provide a basic response when Gemini is unavailable."""
        from src.agent.tools import get_stats, search_segments

        stats = get_stats()

        if "赤" in user_message or "red" in user_message.lower() or "危険" in user_message:
            result = search_segments(level="red")
            segments = result["segments"]
            if segments:
                lines = [f"赤レベル（要対応）のセグメントが {len(segments)} 件あります:\n"]
                for s in segments:
                    lines.append(f"- **{s['segment_id']}**")
                return {"reply": "\n".join(lines), "segment_ids": [s["segment_id"] for s in segments]}

        reply = (
            f"現在、全 {stats['total_segments']} セグメントを監視中です。\n"
            f"- 要対応 (赤): {stats['segments_by_level'].get('red', 0)} 件\n"
            f"- 要注意 (橙): {stats['segments_by_level'].get('orange', 0)} 件\n"
            f"- 経過観察 (黄): {stats['segments_by_level'].get('yellow', 0)} 件\n"
            f"- 異常なし (緑): {stats['segments_by_level'].get('green', 0)} 件\n\n"
            f"※ Gemini API キーが未設定のため、簡易応答モードで動作中です。"
        )
        return {"reply": reply, "segment_ids": []}

    @staticmethod
    def _extract_segment_ids(text: str) -> List[str]:
        """Extract segment IDs like SHINTOMEI_0014 or TOMEI_0087 from text."""
        return list(set(re.findall(r"(?:SHINTOMEI|TOMEI)_\d{4}", text)))
