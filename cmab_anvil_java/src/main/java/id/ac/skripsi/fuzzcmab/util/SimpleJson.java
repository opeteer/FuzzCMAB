package id.ac.skripsi.fuzzcmab.util;

import java.util.Collection;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;

/**
 * Serializer JSON ringan dan bebas dependensi eksternal untuk pembuatan payload JSON-RPC Anvil
 * dan penyimpanan berkas telemetri eksperimen skripsi.
 */
public final class SimpleJson {

    private SimpleJson() {
    }

    public static String toJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeJson(value, sb, 0, false);
        return sb.toString();
    }

    public static String toPrettyJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeJson(value, sb, 0, true);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void writeJson(Object val, StringBuilder sb, int indent, boolean pretty) {
        if (val == null) {
            sb.append("null");
        } else if (val instanceof Boolean || val instanceof Integer || val instanceof Long) {
            sb.append(val.toString());
        } else if (val instanceof Number num) {
            double d = num.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                sb.append("0.0");
            } else {
                sb.append(String.format(Locale.US, "%.6f", d)
                        .replaceAll("(\\.\\d*?[1-9])0+$", "$1")
                        .replaceAll("\\.000000$", ".0"));
            }
        } else if (val instanceof String s) {
            sb.append('"').append(escape(s)).append('"');
        } else if (val instanceof Map<?, ?> map) {
            if (map.isEmpty()) {
                sb.append("{}");
                return;
            }
            sb.append('{');
            if (pretty) {
                sb.append('\n');
            }
            Iterator<? extends Map.Entry<?, ?>> it = map.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<?, ?> entry = it.next();
                if (pretty) {
                    appendIndent(sb, indent + 2);
                }
                sb.append('"').append(escape(String.valueOf(entry.getKey()))).append("\":");
                if (pretty) {
                    sb.append(' ');
                }
                writeJson(entry.getValue(), sb, indent + 2, pretty);
                if (it.hasNext()) {
                    sb.append(',');
                }
                if (pretty) {
                    sb.append('\n');
                }
            }
            if (pretty) {
                appendIndent(sb, indent);
            }
            sb.append('}');
        } else if (val instanceof Collection<?> col) {
            if (col.isEmpty()) {
                sb.append("[]");
                return;
            }
            sb.append('[');
            if (pretty) {
                sb.append('\n');
            }
            Iterator<?> it = col.iterator();
            while (it.hasNext()) {
                if (pretty) {
                    appendIndent(sb, indent + 2);
                }
                writeJson(it.next(), sb, indent + 2, pretty);
                if (it.hasNext()) {
                    sb.append(',');
                }
                if (pretty) {
                    sb.append('\n');
                }
            }
            if (pretty) {
                appendIndent(sb, indent);
            }
            sb.append(']');
        } else if (val instanceof double[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                writeJson(arr[i], sb, indent, false);
            }
            sb.append(']');
        } else if (val instanceof int[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(arr[i]);
            }
            sb.append(']');
        } else if (val instanceof long[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(arr[i]);
            }
            sb.append(']');
        } else {
            sb.append('"').append(escape(val.toString())).append('"');
        }
    }

    private static void appendIndent(StringBuilder sb, int spaces) {
        sb.append(" ".repeat(Math.max(0, spaces)));
    }

    public static String escape(String raw) {
        StringBuilder sb = new StringBuilder(raw.length() + 8);
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.toString();
    }
}
