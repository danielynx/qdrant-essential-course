from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document


load_dotenv('../.env.local')

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

my_dataset = [
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "35",
        "content": """
            Seção III
            Da Presidência  

            Art. 35   São atribuições do Presidente, além das demais expressas neste Regimento:

            I-   quanto às sessões da Assembleia Legislativa:

            a)   presidí-las, abrindo-as, conduzindo-as e encerrando-as, nos termos regimentais;

            b)   suspendê-las sempre que julgar conveniente ao bom andamento técnico ou disciplinar dos trabalhos ou levantá-las, nos termos expressos neste Regimento; 

            c)   manter a ordem e fazer observar o Regimento Interno;

            d)   fazer ler a Ata pelo 2° Secretário, o expediente e as comunicações pelo 1º Secretário;

            e)   conceder a palavra aos Deputados;

            f)   convidar o orador a declarar, quando for o caso, se vai falar a favor ou contra a proposição ou tese em debate;

            g)   interromper o orador que se desviar da questão, falar sobre o vencido ou faltar à consideração devida à Assembleia Legislativa ou a qualquer de seus membros;

            h)   determinar o não registro de discurso ou aparte, pelo serviço de gravação, quando antirregimentais.   (Redação dada pela Res. nº 6699, DOEAL/MT de 06/04/2020)

            h)   determinar o não registro de discurso ou aparte, pela taquigrafia e serviço de gravação, quando anti-regimentais;   (Redação original)

            i)   convidar o Deputado a retirar-se do plenário, quando perturbar a ordem;

            j)   comunicar ao orador que dispõe de três minutos para conclusão do seu pronunciamento, chamar-lhe a atenção ao esgotar-se o tempo a que tem direito, e impedir que, nesse ínterim, sofra ele apartes;

            k)   advertir o orador, ao terminar a hora do Pequeno e do Grande Expediente, que absolutamente não podem sofrer prorrogação;

            l)   decidir soberanamente as questões de ordem e as reclamações, ou delegar a decisão ao Plenário, quando preferir;

            m)   autorizar o Deputado a falar da bancada;

            n)   fazer-se substituir na Presidência, quando tiver que deixar o plenário ou quando tiver que exercer o voto secreto; convocar substitutos eventuais para as Secretarias, na ausência ou impedimento dos Secretários;

            o)   anunciar a Ordem do Dia e o número de Deputados presentes;

            p)   submeter à discussão e votação a matéria a isso destinada;

            q)   estabelecer o ponto da questão sobre o qual deve ser feita a votação e proclamar o seu resultado;

            r)   anunciar, antes do encerramento da sessão, os Deputados que estiveram presentes e os que estiveram ausentes dos seus trabalhos;

            s)   fazer organizar, sob sua responsabilidade e direção, a Ordem do Dia da sessão seguinte e anunciá-la ao término dos trabalhos;

            t)   anunciar, na pauta dos trabalhos, as proposições em condições regimentais de apreciação pelo Plenário;

            u)   convocar sessões extraordinárias, especiais, secretas e solenes, nos termos deste Regimento;

            v)   convocar extraordinariamente a Assembleia Legislativa, nas hipóteses do art. 26;

            w)   promulgar leis nos casos previstos na Constituição Estadual;

            x)   assinar, juntamente com os Secretários, os atos administrativos e as atas das sessões plenárias e das reuniões da Mesa Diretora.

            II-   quanto às proposições:

            a)   distribuir proposições e processos às Comissões;

            b)   deixar de aceitar qualquer proposição que não atenda às exigências regimentais;

            c)   mandar arquivar o relatório ou parecer de Comissão Especial que não haja concluído por projeto;

            d)   determinar a retirada de proposição da Ordem do Dia, nos termos deste Regimento;

            e)   declarar prejudicada qualquer proposição, que assim deva ser considerada, na conformidade regimental;

            f)   despachar os requerimentos, assim verbais como escritos, submetidos à sua apreciação.

            III-   quanto às Comissões:

            a)   nomear, à vista da indicação partidária, os membros efetivos das Comissões e seus suplentes;

            b)   designar, na ausência dos membros das Comissões e seus suplentes, o substituto ocasional observado a filiação partidária;

            c)   declarar a perda de lugar de membro da Comissão, quando incidir no número de faltas previstas no § 2º  do art. 54;

            d)   convocar reunião extraordinária de Comissão para apreciar proposição em regime de urgência;

            e)   nomear Comissão Especial e de Inquérito, nos termos deste Regimento.

            IV-   quanto às reuniões da Mesa Diretora:

            a)   presidí-las;

            b)   tomar parte nas discussões e deliberações, com direito a voto, e assinar as respectivas Atas, Resoluções e Atos;

            c)   distribuir a matéria que dependa de parecer.

            V-   quanto às publicações:

            a)   não permitir a publicação de expressões, conceitos, e discursos infringentes às normas regimentais;

            b)   determinar que as informações oficiais sejam publicadas por extenso, ou apenas em resumo, ou somente referidas na Ata.

            § 1º   Compete também ao Presidente da Assembleia Legislativa:

            I-   dar posse aos Deputados;

            II-   convocar e dar posse aos suplentes;

            III-   presidir as reuniões do Colégio de Líderes, assistido pelo Secretário Parlamentar da Mesa Diretora;   (Redação dada pela Res. nº 7015, DOEAL/MT de 07/06/2021)

            III-   presidir as reuniões do Colégio de Líderes, assistido pelo Consultor Técnico-Jurídico da Mesa Diretora;   (Redação original)

            IV-   assinar a correspondência destinada à Presidência da República, ao Senado Federal, à Câmara dos Deputados, ao Supremo Tribunal Federal, ao Tribunal Superior Eleitoral, aos Ministros de Estado, aos Governadores, aos Tribunais de Justiça, aos Tribunais Regionais Eleitorais, aos Tribunais do Trabalho, aos Tribunais de Contas, às Assembleias Legislativas dos demais Estados e à União Nacional dos Legislativos Estaduais;

            V-   determinar a publicação de atos oficiais do Poder Legislativo no órgão oficial da Assembleia Legislativa ou no Diário Oficial do Estado;

            VI-   dirigir, com suprema autoridade, a polícia da Assembleia Legislativa;

            VII-   zelar pelo prestígio e decoro da Assembleia Legislativa, bem como pela liberdade devida às suas imunidades e demais prerrogativas;

            VIII-   visar a Carteira de Identidade Parlamentar fornecida pela 1" Secretaria da Assembleia Legislativa aos Deputados;

            IX-   assinar ordem de empenho e pagamento juntamente com o 1º Secretário da Assembleia Legislativa.   (Redação dada pela Res. nº 7942, DOEAL/MT de 21/12/2022, em vigor a partir de 01/02/2023)

            IX-   assinar cheques juntamente com o 1º Secretário e o Secretário de Orçamento e Finanças da Assembleia Legislativa.   (Redação original)

            X-   elaborar, anualmente, cronograma para realização de Audiências Públicas, em obediência às determinações do Parágrafo único do art. 48 da Lei Complementar Federal nº 101, de 04 de maio de 2000.

            § 2º   O Presidente não poderá votar, exceto nos casos de empate, de escrutínio secreto e de votação nominal. Em nenhuma hipótese, todavia, votará mais de uma vez para decisão da mesma matéria.

            § 3º   Para tomar parte em qualquer discussão o Presidente deixará a Presidência e não a reassumirá enquanto estiver sob debate a matéria em que interviu.

            § 4º   Em qualquer momento o Presidente poderá, da sua cadeira, fazer ao Plenário comunicação de interesse público ou da Casa.

            § 5º   O Presidente, ou aquele que o substituir, a título de decidir qualquer questão ou quando encaminhar a decisão ao Plenário, jamais poderá fazê-lo em contrariedade à disposição expressa neste Regimento.
        """
    },    
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "46",        
        "content": """
            CAPÍTULO II
            DA POSSE  

            Art. 46   A posse do Deputado, que não se tenha investido do cargo na sessão especial de que tratam os arts. 5º, 8º e 9º, será ato público que se realizará perante a Assembleia Legislativa, em sessão ordinária ou sessão extraordinária, inclusive preparatória, devendo precedê-la a entrega do diploma e da declaração de bens à Mesa Diretora.

            § 1º   Estando a Assembleia Legislativa em recesso, a Mesa Diretora tomará o compromisso e deferirá a posse no gabinete da Presidência.

            § 2º   A apresentação do diploma e da declaração de bens poderá ser feita pelo diplomado, pessoalmente, ou por oficio ao 1º Secretário, como por intermédio do seu Partido ou de qualquer Deputado.

            § 3º   Presente o diplomado, o Presidente designará três Deputados para recebê-lo e introduzi-lo no Plenário das Deliberações, onde, com as formalidades próprias, prestará o compromisso do art. 9º.

            § 4º   Quando forem diversos os Deputados a prestar compromisso, somente um pronunciará a fórmula constante do art. 9° e os demais, um por um, ao serem chamados, dirão: "Assim o prometo".

            § 5º   O Deputado que não tenha sido investido na sessão referida no art. 5o, bem como o suplente convocado, terá, a fim de tomar posse, o prazo de trinta dias, prorrogáveis por mais quinze pela Mesa Diretora, a requerimento escrito do interessado.

            § 6º   Salvo a hipótese do suplente convocado para substituição eventual, perderá o mandato, ou o direito ao seu exercício, o Deputado eleito ou o suplente que deixar de assumir o cargo, sem justificativa aceita por um terço, no mínimo, da Assembleia Legislativa, dentro de quarenta e cinco dias, a contar daquele em que lhe foi o mesmo posto à disposição.

            § 7º   Na hipótese de ocorrência de vaga no período de recesso parlamentar, a posse do suplente far-se-á perante o Presidente da Assembleia Legislativa, em ato público realizado no seu gabinete, observado o disposto no art. 9º
        """
    }, 
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "51",
        "content": """
            Art. 51   A convocação de suplente, em caso de vacância que a autorize, será imediata à abertura da vaga.
        """
    },    
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "55",
        "content": """
            CAPÍTULO VI
            DA CONVOCAÇÃO DE SUPLENTE  

            Art. 55   A Mesa Diretora convocará, no prazo de quarenta e oito horas, o suplente de Deputado, nos casos de:

            I-   ocorrência de vaga;

            II-   licença do titular, nos casos previstos no art. 52, incisos IV, VI e VIII;   (Redação dada pela Res. nº 6812, DOEAL/MT de 13/08/2020)

            II-   Licença do Titular, previsto no Art. 52, IV e VI;   (Redação dada pela Res. nº 845, D.O. de 13/03/2008)

            II-   licença do titular, prevista no art. 52, IV;   (Redação original)

            III-   licença médica, prevista no art. 52, V, desde que ultrapasse 120 dias.

            § 1º   O Deputado que se licenciar pelo inciso III, com assunção de suplente, poderá reassumir o mandato antes de findo o prazo da licença ou de suas prorrogações, desde que apresente atestado médico informando o restabelecimento de sua saúde.

            § 2º   Assiste ao suplente que for convocado o direito de se declarar impossibilitado de assumir o exercício do mandato, dando ciência por escrito à Mesa Diretora, que convocará o suplente imediato, após registro nos Anais da Casa.
        """
    }, 
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "56",
        "content": """
            CAPÍTULO VI
            DA CONVOCAÇÃO DE SUPLENTE  

            Art. 56   Ocorrendo vaga e não havendo suplente, far-se-á eleição para preenchê-la, se faltarem mais de quinze meses para o término do mandato.
        """
    },   
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "57",
        "content": """
            CAPÍTULO VI
            DA CONVOCAÇÃO DE SUPLENTE  

            Art. 57   O suplente de Deputado, quando convocado em caráter de substituição por tempo determinado, não poderá ser escolhido para os cargos da Mesa Diretora, Presidente ou Vice-Presidente de Comissão.   (Redação dada pela Res. nº 1088, D.O. de 19/02/2009)

            Art. 57   O suplente de Deputado, quando convocado em caráter de substituição, não poderá ser escolhido para os cargos da Mesa Diretora, Presidente ou Vice-Presidente de Comissão.   (Redação original)
        """
    },  
    {
        "law": "Resolução nº 677 de 20 de dezembro de 2006",
        "law_number": "677",
        "year": "2006",
        "article": "66",
        "content": """
            CAPÍTULO X
            DO NOME PARLAMENTAR  

            Art. 66   Ao assumir o exercício do mandato o Deputado ou suplente convocado escolherá o nome parlamentar com que deverá figurar nas publicações ou registros da Casa.

            § 1º   O nome parlamentar não constará de mais de três palavras, não computadas, nesse número, as preposições ou conjunções, bem assim os termos Filho, Júnior, Neto, Sobrinho ou semelhantes.

            § 2º   Ocorrendo coincidência de nomes parlamentares, sem entendimento entre os interessados, para dirimir a duplicidade optará preferencialmente o Deputado mais antigo, ou, não existindo, o mais idoso.

            § 3º   A Carteira de Identidade Parlamentar registrará por inteiro o nome do Deputado, consignando-lhe, todavia, em maiúscula, os elementos constitutivos do nome parlamentar.

            § 4º   Ao Deputado é lícito, a qualquer tempo, mudar seu nome parlamentar, através de comunicado escrito à Mesa Diretora.
        """
    },                      
    {
        "law": "Constituição nº de 5 de outubro de 1989",
        "law_number": "",
        "year": "1989",
        "article": "32",
        "content": """
            Seção III
            Dos Deputados Estaduais   

            Art. 32   Não perderá o mandato o Deputado Estadual:

            I-   investido no cargo de Ministro de Estado, Secretário de Estado e de Prefeitura da Capital;

            II-   licenciado pela Assembleia Legislativa por motivo de doença ou para tratar, sem remuneração, de interesse particular, desde que, neste caso, o afastamento não ultrapasse 180 (cento e oitenta) dias por Sessão Legislativa.   (Redação dada pela EC nº 68, D.O. de 24/10/2014, com efeitos a partir de 16/10/2014)

            II-   licenciado pela Assembleia Legislativa por motivo de doença, ou para tratar, sem remuneração, de interesse particular, desde que, neste caso, o afastamento não ultrapasse cento e vinte dias por sessão legislativa.   (Redação original)

            § 1º   O suplente será convocado nos casos de vaga, de investidura em funções previstas neste artigo ou de licença superior a cento e vinte dias.

            § 2º   Ocorrendo vaga e não havendo suplente, far-se-á eleição para preenchê-la se faltarem mais de quinze meses para o término do mandato.

            § 3º   Na hipótese do inciso I, o Deputado Estadual poderá optar pela remuneração do mandato.
        """
    },
]

def fixed_size_chunks(text, chunk_size=100, overlap=20):
    """Split text into fixed-size chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Only add non-empty chunks
            chunks.append(' '.join(chunk_words))
    
    return chunks

def sentence_chunks(text, max_sentences=3):
    """Group sentences into chunks"""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if chunk_sentences:
            chunks.append('. '.join(chunk_sentences) + '.')
    
    return chunks

def paragraph_chunks(text):
    """Split by paragraphs or double line breaks"""
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return chunks if chunks else [text]  # Fallback to full text
    
def semantic_chunks(text):
    document = Document(text=text)

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    nodes = semantic_splitter.get_nodes_from_documents([document])  # Pass list of Document objects
    return [n.text for n in nodes]

collection_name = "day1_semantic_search"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

# Create a collection with three named vectors
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "fixed": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "sentence": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "paragraph": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "semantic": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)

# Index fields for filtering (more on this on day 2)
client.create_payload_index(
    collection_name=collection_name,
    field_name="chunk_strategy",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="article",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# Process and upload data
points = []
point_id = 0

for item in my_dataset:
    content = item["content"]

    # Process with each chunking strategy
    strategies = {
        "fixed": fixed_size_chunks(content),
        "sentence": sentence_chunks(content),
        "paragraph": paragraph_chunks(content),
        "semantic": semantic_chunks(content)
    }

    for strategy_name, chunks in strategies.items():
        for chunk_idx, chunk in enumerate(chunks):
            # Create vectors for this chunk
            vectors = {strategy_name: encoder.encode(chunk).tolist()}

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={
                        **item,  # Include all original metadata
                        "chunk": chunk,
                        "chunk_strategy": strategy_name,
                        "chunk_index": chunk_idx,
                    },
                )
            )
            point_id += 1

client.upload_points(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} chunks across three strategies")


def compare_search_results(query):
    """Compare search results across all chunking strategies"""
    print(f"Query: '{query}'\n")

    for strategy in ["fixed", "sentence", "paragraph", "semantic"]:
        results = client.query_points(
            collection_name=collection_name,
            query=encoder.encode(query).tolist(),
            using=strategy,
            limit=3,
        )

        print(f"--- {strategy.upper()} CHUNKING ---")
        for i, point in enumerate(results.points, 1):
            print(f"{i}. {point.payload['law']}")
            print(f"   Score: {point.score:.3f}")
            print(f"   Article: {point.payload['article']}")
            print(f"   Chunk: {point.payload['chunk']}")
        print()

# Test with domain-specific queries
test_queries = [
    "Quando a Mesa Diretora irá convocar o suplente de deputado?",
    "Quem pode convocar o suplente de deputado?",
    "Como o suplente de deputado pode perder o mandato?",
]

for query in test_queries:
    compare_search_results(query)


def analyze_chunking_effectiveness():
    """Analyze which chunking strategy works best for your domain"""

    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 40)

    # Get chunk statistics for each strategy
    for strategy in ["fixed", "sentence", "paragraph", "semantic"]:
        # Count chunks per strategy
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_strategy", match=models.MatchValue(value=strategy)
                    )
                ]
            ),
            limit=100,
        )

        chunks = results[0]
        chunk_sizes = [len(chunk.payload["chunk"]) for chunk in chunks]

        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
        print(f"  Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")


analyze_chunking_effectiveness()