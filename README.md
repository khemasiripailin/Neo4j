# Neo4j GDS Network Visualizer: Centrality + K-means + IMDb

Streamlit app สำหรับงาน Social Network / Graph Data Science โดยคำนวณผ่าน Neo4j GDS จริง

## สิ่งที่มีในเว็บ

1. Network Visualizer จาก `edges_rows.csv` และ `favorites_rows.csv`
2. Centrality: Betweenness, Bridges, Closeness, Degree, Eigenvector, PageRank
3. Ex.3 K-means community detection on `edges_rows.csv`
4. Ex.4 K-means community detection on `karate_club`
5. Ex.5 K-means community detection on IMDb sample graph โดยใช้คำสั่ง `G = gds.graph.load_imdb()`
6. Cypher + Notes อธิบาย query หลักว่าใช้สื่ออะไร

## จุดที่แก้เพิ่ม

- เพิ่มอาหารจาก `favorites_rows.csv` เป็น node ประเภท Food และ relationship `likes`
- กราฟมี Legend
- กราฟใช้สีไล่ระดับตามค่า centrality/community
- กราฟแท่งมีตัวเลขบนแท่งเพื่ออ่านง่าย
- IMDb ใช้ `graphdatascience` package เพื่อโหลด sample graph ด้วย `gds.graph.load_imdb()`

## วิธีรัน local

ก่อนรันให้ Neo4j Docker ที่มี GDS plugin เปิดอยู่แล้ว เช่น login ได้ที่ `http://localhost:7474/`

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

ค่าที่ใช้ในหน้าเว็บ:

```text
Neo4j URI: bolt://localhost:7687
Username: neo4j
Password: 12345678
Database: neo4j
```

## หมายเหตุ Streamlit Cloud

ถ้า deploy บน Streamlit Cloud จะใช้ `bolt://localhost:7687` ไม่ได้ ต้องใช้ Neo4j Aura หรือ Neo4j server ที่เข้าจากอินเทอร์เน็ตได้
