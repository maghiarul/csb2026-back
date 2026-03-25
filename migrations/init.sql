CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('user', 'admin');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'image_status') THEN
        CREATE TYPE image_status AS ENUM ('pending', 'approved', 'rejected');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    role user_role NOT NULL DEFAULT 'user',
    username TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE IF NOT EXISTS public.plants (
    id BIGSERIAL PRIMARY KEY,
    name_ro TEXT NOT NULL UNIQUE,
    name_latin TEXT NOT NULL,
    usable_parts TEXT,
    health_benefits TEXT,
    contraindications TEXT,
    description TEXT,
    image_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE IF NOT EXISTS public.points_of_interest (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    plant_id BIGINT NOT NULL REFERENCES public.plants(id) ON DELETE RESTRICT,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE IF NOT EXISTS public.poi_images (
    id BIGSERIAL PRIMARY KEY,
    poi_id BIGINT NOT NULL REFERENCES public.points_of_interest(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    status image_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_poi_location_gist ON public.points_of_interest USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_poi_plant_id ON public.points_of_interest (plant_id);
CREATE INDEX IF NOT EXISTS idx_poi_images_status ON public.poi_images (status);
CREATE INDEX IF NOT EXISTS idx_poi_images_poi_id ON public.poi_images (poi_id);

CREATE OR REPLACE FUNCTION public.create_poi_with_image(
    p_user_id UUID,
    p_plant_id BIGINT,
    p_lat DOUBLE PRECISION,
    p_lng DOUBLE PRECISION,
    p_comment TEXT,
    p_image_url TEXT
)
RETURNS TABLE (
    poi_id BIGINT,
    image_id BIGINT,
    user_id UUID,
    plant_id BIGINT,
    comment TEXT,
    image_status image_status,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_poi_id BIGINT;
    v_image_id BIGINT;
    v_created_at TIMESTAMPTZ;
BEGIN
    INSERT INTO public.points_of_interest (user_id, plant_id, location, comment)
    VALUES (
        p_user_id,
        p_plant_id,
        ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::geography,
        p_comment
    )
    RETURNING id, points_of_interest.created_at INTO v_poi_id, v_created_at;

    INSERT INTO public.poi_images (poi_id, user_id, image_url, status)
    VALUES (v_poi_id, p_user_id, p_image_url, 'pending')
    RETURNING id INTO v_image_id;

    RETURN QUERY
    SELECT
        v_poi_id,
        v_image_id,
        p_user_id,
        p_plant_id,
        p_comment,
        'pending'::image_status,
        v_created_at;
END;
$$;

CREATE OR REPLACE FUNCTION public.get_approved_poi(
    p_plant_id BIGINT DEFAULT NULL,
    p_lat DOUBLE PRECISION DEFAULT NULL,
    p_lng DOUBLE PRECISION DEFAULT NULL,
    p_radius_km DOUBLE PRECISION DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    user_id UUID,
    plant_id BIGINT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    comment TEXT,
    created_at TIMESTAMPTZ,
    distance_km DOUBLE PRECISION,
    image_path TEXT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        poi.id,
        poi.user_id,
        poi.plant_id,
        ST_Y(poi.location::geometry) AS latitude,
        ST_X(poi.location::geometry) AS longitude,
        poi.comment,
        poi.created_at,
        CASE
            WHEN p_lat IS NOT NULL AND p_lng IS NOT NULL THEN
                ST_Distance(
                    poi.location,
                    ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::geography
                ) / 1000.0
            ELSE NULL
        END AS distance_km,
        approved_image.image_url AS image_path
    FROM public.points_of_interest AS poi
    JOIN LATERAL (
        SELECT img.image_url
        FROM public.poi_images AS img
        WHERE img.poi_id = poi.id
          AND img.status = 'approved'
        ORDER BY img.created_at DESC
        LIMIT 1
    ) AS approved_image ON TRUE
    WHERE (p_plant_id IS NULL OR poi.plant_id = p_plant_id)
      AND (
        p_lat IS NULL OR p_lng IS NULL OR p_radius_km IS NULL OR ST_DWithin(
            poi.location,
            ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::geography,
            p_radius_km * 1000.0
        )
      )
    ORDER BY poi.created_at DESC;
$$;

CREATE OR REPLACE FUNCTION public.get_poi_detail(p_poi_id BIGINT)
RETURNS TABLE (
    id BIGINT,
    user_id UUID,
    plant_id BIGINT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    comment TEXT,
    created_at TIMESTAMPTZ,
    name_ro TEXT,
    name_latin TEXT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        poi.id,
        poi.user_id,
        poi.plant_id,
        ST_Y(poi.location::geometry) AS latitude,
        ST_X(poi.location::geometry) AS longitude,
        poi.comment,
        poi.created_at,
        p.name_ro,
        p.name_latin
    FROM public.points_of_interest AS poi
    JOIN public.plants AS p ON p.id = poi.plant_id
    WHERE poi.id = p_poi_id
    LIMIT 1;
$$;

INSERT INTO public.plants (
    name_ro,
    name_latin,
    usable_parts,
    health_benefits,
    contraindications,
    description,
    image_url
)
VALUES
    ('Musetel', 'Matricaria chamomilla', 'Flori', 'Calmeaza tractul digestiv, antiinflamator, sustine somnul', 'Alergie la Asteraceae, precautie in sarcina', 'Planta frecventa in zonele de campie din Moldova, folosita in infuzii pentru digestie si calmare.', NULL),
    ('Sunatoare', 'Hypericum perforatum', 'Parti aeriene inflorite', 'Sustinere emotionala usoara, cicatrizant, antiinflamator', 'Interactiuni medicamentoase multiple, fotosensibilitate', 'Creste pe margini de drum si pajisti insorite; utilizata traditional pentru stari anxioase usoare.', NULL),
    ('Pelin', 'Artemisia absinthium', 'Parti aeriene', 'Stimuleaza digestia, sustine functia biliara', 'Evitat in sarcina, doze mari pot fi neurotoxice', 'Planta amara specifica terenurilor uscate, folosita in cure digestive scurte.', NULL),
    ('Coada-soricelului', 'Achillea millefolium', 'Parti aeriene inflorite', 'Antispastic, hemostatic usor, sustine digestia', 'Alergie la Asteraceae', 'Planta comuna pe fanete si margini de camp, utilizata in ceaiuri si comprese.', NULL),
    ('Patlagina', 'Plantago major', 'Frunze', 'Calmeaza tusea, emolient, antiinflamator local', 'Rar alergii de contact', 'Apare frecvent pe poteci si terenuri tasate; folosita pentru siropuri si aplicatii locale.', NULL),
    ('Urzica', 'Urtica dioica', 'Frunze, varfuri tinere, radacina', 'Remineralizant, diuretic usor, sustine metabolismul', 'Precautie in insuficienta renala severa', 'Planta raspandita in zone umede, folosita alimentar si medicinal primavara.', NULL),
    ('Menta', 'Mentha piperita', 'Frunze', 'Antispastic digestiv, reduce balonarea, racoritor', 'Poate agrava refluxul gastroesofagian', 'Cultivata si spontana in gradini umede; utilizata in infuzii si inhalatii.', NULL),
    ('Salvie', 'Salvia officinalis', 'Frunze', 'Antiseptic oral, reduce transpiratia excesiva, antiinflamator', 'Evitata in sarcina si alaptare in doze mari', 'Planta aromatica adaptata si in gradini locale, folosita in gargara si ceai.', NULL),
    ('Tei', 'Tilia cordata', 'Flori', 'Sedativ bland, sudorific, calmant respirator', 'Rar hipotensiune la consum excesiv', 'Arbore prezent in localitati si paduri de foioase; florile sunt recoltate pentru ceai.', NULL),
    ('Soc', 'Sambucus nigra', 'Flori, fructe coapte', 'Sudorific, sustine imunitatea, antioxidant', 'Partile crude necoapte pot fi toxice', 'Arbust comun in liziere; florile se folosesc in infuzii, fructele in preparate termice.', NULL),
    ('Rostopasca', 'Chelidonium majus', 'Parti aeriene, latex', 'Utilizare traditionala pentru aplicatii locale', 'Potential hepatotoxic intern, utilizare interna doar cu aviz medical', 'Planta de umbra partiala, cunoscuta pentru latexul portocaliu.', NULL),
    ('Cicoare', 'Cichorium intybus', 'Radacina, parti aeriene', 'Sustine digestia, prebiotic natural', 'Alergie la Asteraceae', 'Planta cu flori albastre, frecventa pe margini de drum si terenuri deschise.', NULL),
    ('Maces', 'Rosa canina', 'Fructe', 'Bogata in vitamina C, antioxidant, sustine imunitatea', 'Poate irita gastric in doze mari', 'Arbust spontan in zone de deal si campie, fructele sunt recoltate toamna.', NULL),
    ('Traista-ciobanului', 'Capsella bursa-pastoris', 'Parti aeriene', 'Hemostatic traditional, diuretic usor', 'Precautie in sarcina', 'Buruiana comuna in culturi si la marginea drumurilor, folosita in fitoterapia traditionala.', NULL),
    ('Lumanarica', 'Verbascum thapsus', 'Flori, frunze', 'Calmeaza tusea, emolient respirator', 'Filtrare atenta a infuziei pentru a evita peri iritanti', 'Planta bienala in zone insorite si uscate, frecvent folosita in ceaiuri pentru gat.', NULL)
ON CONFLICT (name_ro) DO NOTHING;
