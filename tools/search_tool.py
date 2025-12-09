from ddgs import DDGS

from tools import Tool

class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "- Use this tool to search the web."
        "- Parameters:"
        "  - \"query\": the search query string"
        )

    def __call__(self, query: str, max_results: int = 5) -> str:
        """
        질문에 대한 검색 결과를 반환합니다.

        args:
            query(str): 검색할 키워드
            max_results(int): 최대 검색 결과 수

        returns:
            str: 제목과 내용, URL을 검색결과에 맞게 순차적으로 반환
        """

        results = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            return f"[Web search error] Error occurred during search: {e}"
        
        if not results:
            return f"[Web search] No results found for query: '{query}'"
        
        for i, item in enumerate(results):
            title = item.get("title", "")
            title = title.replace("<b>", "").replace("</b>", "")
            body = item.get("body", "")
            url = item.get("href", "")
            results[i] = f"[{i+1}] Title: {title}\n context: {body}\n URL: {url}\n"

        return "\n\n".join(results)