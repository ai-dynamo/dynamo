"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { schemaBaseURL, schemaSources } from "./schemaSources.generated";
import { KubeSchemaDoc } from "./KubeSchemaDoc";
import type { KubeSchemaDocument } from "./KubeSchemaDoc";

function schemaURL(fileName: string) {
  return `${schemaBaseURL.replace(/\/$/, "")}/${fileName}`;
}

function fetchSchema(fileName: string) {
  return fetch(schemaURL(fileName)).then((response) => {
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json() as Promise<KubeSchemaDocument>;
  });
}

export function LazyKubeSchemaDoc({ name, filtering = true }: { name: string; filtering?: boolean }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<KubeSchemaDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const schemaGenerationRef = useRef(0);
  const loadingInitialRef = useRef(false);
  const fullLoadRef = useRef<Promise<KubeSchemaDocument> | null>(null);

  useEffect(() => {
    const element = rootRef.current;
    if (!element || typeof IntersectionObserver === "undefined") {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      setIsVisible(entries.some((entry) => entry.isIntersecting));
    }, { rootMargin: "160px 0px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    schemaGenerationRef.current += 1;
    setData(null);
    setError(null);
    loadingInitialRef.current = false;
    fullLoadRef.current = null;
  }, [name]);

  useEffect(() => {
    let cancelled = false;
    const source = schemaSources[name];

    if (!source) {
      setError(`Unknown schema document: ${name}`);
      return () => {
        cancelled = true;
      };
    }
    if (!isVisible || data || loadingInitialRef.current) {
      return () => {
        cancelled = true;
      };
    }

    loadingInitialRef.current = true;
    const generation = schemaGenerationRef.current;
    fetchSchema(source.initial)
      .then((payload) => {
        if (!cancelled && generation === schemaGenerationRef.current) {
          setData(payload);
        }
      })
      .catch((loadError: unknown) => {
        if (!cancelled && generation === schemaGenerationRef.current) {
          loadingInitialRef.current = false;
          setError(loadError instanceof Error ? loadError.message : String(loadError));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [data, isVisible, name]);

  const loadFull = useCallback(() => {
    const source = schemaSources[name]?.full;
    if (!source || data?.complete) {
      return false;
    }
    if (fullLoadRef.current) {
      return fullLoadRef.current;
    }

    const generation = schemaGenerationRef.current;
    const promise = fetchSchema(source)
      .then((next) => {
        if (generation !== schemaGenerationRef.current) {
          throw new Error("schema request superseded");
        }
        setData(next);
        return next;
      })
      .catch((loadError: unknown) => {
        if (generation === schemaGenerationRef.current) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
          fullLoadRef.current = null;
        }
        throw loadError;
      });
    fullLoadRef.current = promise;
    return promise;
  }, [data?.complete, name]);

  return (
    <div ref={rootRef} className="kdoc-fern-lazy-frame">
      {error ? (
        <div className="kdoc-fern-lazy kdoc-fern-lazy-error">Schema failed to load: {error}</div>
      ) : data ? (
        <KubeSchemaDoc data={data} filtering={filtering} loadFullSchema={loadFull} />
      ) : (
        <div className="kdoc-fern-lazy">Loading schema...</div>
      )}
    </div>
  );
}

export default LazyKubeSchemaDoc;
